# @noautodeps
import hashlib
import os
from collections import Counter

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from fbsearch.model.llama import FBSearchTransformer
from fbsearch.model.prompt import PromptFormat
from torch.utils.data import DataLoader, Dataset

DEBUG = os.environ.get("DEBUG", False)
plt.rcParams["font.family"] = "DejaVu Sans"


class CorpusDataset(Dataset):
    def __init__(self, corpus: pd.DataFrame):
        super().__init__()
        if not corpus.index.name == "doc_id":
            try:
                corpus = corpus.set_index("doc_id")
            except KeyError:
                raise KeyError("Corpus must have a column named 'doc_id'")
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, i):
        doc_id = int(self.corpus.index[i])
        content = self.corpus["content"][doc_id]
        return {
            "doc_id": int(doc_id),
            "content": content,
        }


def corpus_collate_fn(batch):
    doc_ids = [item["doc_id"] for item in batch]
    texts = [item["content"] for item in batch]
    return {"doc_id": doc_ids, "content": texts}


class QueryDataset(Dataset):
    def __init__(self, queries: pd.DataFrame):
        super().__init__()
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        citations = self.queries["citations"][i]
        query = self.queries["query"][i]
        return {
            "doc_id": citations,
            "query": query,
        }


def query_collate_fn(batch):
    queries = [item["query"] for item in batch]
    citations = [item["doc_id"] for item in batch]

    return {"query": queries, "doc_id": citations}


class Query2DocumentDataset(Dataset):
    def __init__(self, queries: pd.DataFrame, corpus: pd.DataFrame):
        super().__init__()
        self.queries = queries
        if not corpus.index.name == "doc_id":
            try:
                corpus = corpus.set_index("doc_id")
            except KeyError:
                raise KeyError("Corpus must have a column named 'doc_id'")
        self.corpus = corpus

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        citations = self.queries["citations"][i]
        query = self.queries["query"][i]

        docs = []
        for doc_id in citations:
            docs.append(self.corpus.loc[doc_id, "content"])
        n_docs = len(docs)

        if n_docs == 0:
            return {}
        else:
            queries = [query] * n_docs

        return {
            "doc_id": citations,
            "query": queries,
            "content": docs,
        }


def query2doc_collate_fn(batch):
    """
    This handles the case where within a batch, queries map to variable number of documents.
    For example, query1 maps to doc1, and query2 maps to doc2, doc3.
    Then, the batch will be like: (query1, doc1), (query2, doc2), (query2, doc3).
    """
    batch = [item for item in batch if item]
    if not batch:
        return {}
    doc_ids = sum([item["doc_id"] for item in batch], [])
    queries = sum([item["query"] for item in batch], [])
    texts = sum([item["content"] for item in batch], [])
    return {
        "doc_id": doc_ids,
        "query": queries,
        "content": texts,
    }


def get_retrieval_datasets(
    corpus_path,
    query_train_path,
    query_dev_path,
):
    corpus = pd.read_json(corpus_path, lines=True)
    query_train = pd.read_json(query_train_path, lines=True)
    query_dev = pd.read_json(query_dev_path, lines=True)

    corpus_dataset = CorpusDataset(corpus)
    query2doc_dataset = Query2DocumentDataset(query_train, corpus)

    query_train_dataset = QueryDataset(query_train)
    query_dev_dataset = QueryDataset(query_dev)

    return {
        # Corpus
        "corpus_dataset": corpus_dataset,
        "corpus_collate_fn": corpus_collate_fn,
        # Query to document for training
        "query2doc_dataset": query2doc_dataset,
        "query2doc_collate_fn": query2doc_collate_fn,
        # Query for training
        "query_train_dataset": query_train_dataset,
        # Query for validation
        "query_dev_dataset": query_dev_dataset,
        # Collate functions for query
        "query_collate_fn": query_collate_fn,
    }


def get_sample_datasets(file_dir=None, query_type="many2many", batch_size=1024):
    if file_dir is None:
        file_dir = os.path.dirname(__file__)
    corpus_path = os.path.join(file_dir, "examples/retrieval/corpus.jsonl")
    query_path = os.path.join(file_dir, f"examples/retrieval/{query_type}.jsonl")

    corpus = pd.read_json(corpus_path, lines=True)
    queries = pd.read_json(query_path, lines=True)

    corpus_dataset = CorpusDataset(corpus)
    corpus_dataloader = DataLoader(
        corpus_dataset, batch_size=batch_size, collate_fn=corpus_collate_fn
    )

    query_dataset = QueryDataset(queries)
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, collate_fn=query_collate_fn
    )

    query2doc_dataset = Query2DocumentDataset(queries, corpus)
    query2doc_dataloader = DataLoader(
        query2doc_dataset, batch_size=batch_size, collate_fn=query2doc_collate_fn
    )

    datas = {
        "corpus": next(iter(corpus_dataloader)),
        "query": next(iter(query_dataloader)),
        "query2doc": next(iter(query2doc_dataloader)),
        "corpus_dataset": corpus_dataset,
        "query_dataset": query_dataset,
        "query2doc_dataset": query2doc_dataset,
        "corpus_collate_fn": corpus_collate_fn,
        "query_collate_fn": query_collate_fn,
        "query2doc_collate_fn": query2doc_collate_fn,
    }
    return datas


class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.doc_ids = set()


class PrefixTreeStore:
    def __init__(
        self,
        transformer: FBSearchTransformer = None,
        insertion_depth: int = 5,
    ):
        self.transformer = transformer
        self.root = PrefixTreeNode()
        self.insertion_depth = insertion_depth

        # {0: {"content": "This is a test document.", "genx": [[0, 1, 2], [1, 2, 3]]}, ...}
        self.data_store = {}

    def insert(self, docs, genxs=None, tokens=None):
        doc_ids: list[int] = docs["doc_id"]
        contents: list[str] = docs["content"]
        if genxs is None and tokens is None:
            genxs, tokens = self.transformer.index_doc(contents)
        if isinstance(genxs, torch.Tensor):
            genxs = genxs.cpu().numpy().astype(int)  # 2D numpy array

        num_beams_doc = self.transformer.num_beams_doc

        for i in range(len(doc_ids)):
            doc_id = doc_ids[i]
            content = contents[i]
            start = i * num_beams_doc
            end = (i + 1) * num_beams_doc
            genx_beams = genxs[start:end]
            token_beams = tokens[start:end]

            # Put into the data store
            self.data_store[doc_id] = {
                "content": content,
                "genx": genx_beams,
                "token": token_beams,
            }

            # Insert into the prefix tree
            for genx in genx_beams:
                self.traverse_and_insert(genx, doc_id)

    def traverse_and_insert(self, genx: np.array, doc_id: int):
        assert len(genx.shape) == 1
        node = self.root
        depth = 0
        for idx in genx:
            if idx not in node.children:
                node.children[idx] = PrefixTreeNode()
            node = node.children[idx]
            depth += 1
            if depth >= self.insertion_depth:
                node.doc_ids.add(doc_id)

    def query(self, queries: list[str], genxs=None, tokens=None):
        if genxs is None and tokens is None:
            genxs, tokens = self.transformer.index_query(queries)
        if isinstance(genxs, torch.Tensor):
            genxs = genxs.cpu().numpy().astype(int)  # 2D numpy array

        num_beams_query = self.transformer.num_beams_query

        results = []
        for i in range(len(queries)):
            genx = genxs[i]

            start = i * num_beams_query
            end = (i + 1) * num_beams_query
            genx_beams = genxs[start:end]
            token_beams = tokens[start:end]

            result = {}
            result["found"] = []
            for genx in genx_beams:
                depth, found_doc_ids = self.traverse_and_search(genx)
                result["found"].append({"depth": depth, "doc_ids": found_doc_ids})

            result["genx"] = genx_beams
            result["token"] = token_beams
            results.append(result)
        return results

    @staticmethod
    def results2metrics(doc_ids: list[list[int]], results: list[dict]):
        metrics = {}

        total_pred = 0
        total_pred_correct = 0
        total_labels = 0

        for idx in range(len(results)):
            result = results[idx]
            labels = doc_ids[idx]
            found = result["found"]
            doc_ids_pred = set()
            for item in found:
                doc_ids_pred.update(item["doc_ids"])

            total_pred += len(doc_ids_pred)

            doc_ids_true: list[int] = labels
            total_labels += len(doc_ids_true)

            n_pred_correct = len(doc_ids_pred.intersection(set(doc_ids_true)))
            total_pred_correct += n_pred_correct
            if DEBUG:
                print("doc_ids_pred:", doc_ids_pred)
                print("doc_ids_true:", doc_ids_true)
                print("pred_correct:", n_pred_correct)

        metrics["precision"] = 0
        metrics["recall"] = 0
        metrics["f1"] = 0

        if total_pred > 0:
            metrics["precision"] = total_pred_correct / total_pred
        if total_labels > 0:
            metrics["recall"] = total_pred_correct / total_labels
        if total_pred_correct > 0:
            metrics["f1"] = (
                2
                * metrics["precision"]
                * metrics["recall"]
                / (metrics["precision"] + metrics["recall"])
            )

        return results, metrics

    def eval_with_query(
        self,
        queries: list[str],
        doc_ids: list[list[int]],
        genxs=None,
        tokens=None,
    ):
        results = self.query(queries, genxs, tokens)
        return self.results2metrics(doc_ids, results)

    def traverse_and_search(self, genx: np.array):
        assert len(genx.shape) == 1
        node: PrefixTreeNode = self.root
        found = True
        depth = 0

        for idx in genx:
            if (idx not in node.children) and (depth < self.insertion_depth):
                found = False
                break
            if (idx not in node.children) and (depth >= self.insertion_depth):
                break
            node = node.children[idx]
            depth += 1

        if found:
            return depth, node.doc_ids
        return -1, set({})

    def plot_token_frequencies(
        self,
        figsize=(12, 8),
        show_top_n=10,
        save_path=None,
    ):
        # Extract all tokens from the data store
        all_tokens = []
        for result_dict in self.data_store.values():
            for token in result_dict["token"]:
                all_tokens.append(token)

        if not all_tokens:
            print("No data found in the input dictionary.")
            fig = plt.figure(figsize=figsize)
            return None, fig

        # Count frequencies of each unique token
        token_frequencies = Counter(all_tokens)

        # Count the frequency of frequencies (meta-frequency distribution)
        frequency_distribution = Counter(token_frequencies.values())

        # Create the figure and axes
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # Plot 1: Distribution of frequency counts
        frequency_values = sorted(frequency_distribution.keys())
        frequency_counts = [frequency_distribution[freq] for freq in frequency_values]

        axes[0].bar(
            frequency_values,
            frequency_counts,
            color="lightblue",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_xlabel("Frequency Count", fontsize=24)
        axes[0].set_ylabel("Number of Tokens", fontsize=24)
        axes[0].set_title("Distribution of Token Frequencies", fontsize=28)
        axes[0].grid(axis="y", linestyle="--", alpha=0.6)
        axes[0].tick_params(axis="x", labelsize=20)
        axes[0].tick_params(axis="y", labelsize=20)

        # Plot 2: Histogram of all frequency values
        all_frequency_values = list(token_frequencies.values())
        max_frequency = max(all_frequency_values)
        axes[1].hist(
            all_frequency_values,
            bins=range(1, max_frequency + 2),
            rwidth=0.8,
            align="left",
            color="lightcoral",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_xlabel("Frequency", fontsize=24)
        axes[1].set_ylabel("Count", fontsize=24)
        axes[1].set_title("Histogram of Token Frequencies", fontsize=28)
        axes[1].grid(axis="y", linestyle="--", alpha=0.6)
        axes[1].tick_params(axis="x", labelsize=20)
        axes[1].tick_params(axis="y", labelsize=20)

        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        # Calculate statistics
        stats = {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(token_frequencies),
            "tokens_appearing_once": frequency_distribution.get(1, 0),
            "tokens_appearing_multiple": sum(
                count for freq, count in frequency_distribution.items() if freq > 1
            ),
            "max_frequency": max(all_frequency_values),
            "frequency_distribution": dict(frequency_distribution),
        }

        # Print statistics
        print("Summary Statistics:")
        print(f"Total number of tokens: {stats['total_tokens']}")
        print(f"Number of unique tokens: {stats['unique_tokens']}")
        print(f"Tokens appearing once: {stats['tokens_appearing_once']}")
        print(f"Tokens appearing more than once: {stats['tokens_appearing_multiple']}")
        print(f"Maximum frequency: {stats['max_frequency']}")

        # Show top frequent tokens
        if len(token_frequencies) <= show_top_n * 2:
            print("\nAll unique tokens and their frequencies:")
            for i, (unique_token, frequency_count) in enumerate(
                sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True), 1
            ):
                print(f"{i:2d}: {unique_token} appears {frequency_count} time(s)")
        else:
            print(f"\nTop {show_top_n} most frequent tokens:")
            for i, (unique_token, frequency_count) in enumerate(
                sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)[
                    :show_top_n
                ],
                1,
            ):
                print(f"{i:2d}: {unique_token} appears {frequency_count} time(s)")

        print("\nFrequency distribution:")
        for frequency_value in sorted(frequency_distribution.keys()):
            token_count = frequency_distribution[frequency_value]
            print(f"{token_count} tokens appear exactly {frequency_value} time(s)")

        return stats, fig


def text_to_int_list(texts, n=5, num_range=10, num_beams=1):
    if not isinstance(texts, list):
        texts = [texts]

    results = []
    for text in texts:
        beams = []
        for beam_idx in range(num_beams):
            beam_text = f"{text}_beam_{beam_idx}"
            # Add beam index to create different hashes for each beam
            beam_text = f"{text}_beam_{beam_idx}"
            # Hash the input text using SHA256
            digest = hashlib.sha256(beam_text.encode("utf-8")).digest()
            # Convert the first n bytes to integers in the desired range
            hashcode = [digest[i] % num_range for i in range(n)]
            beams.append(hashcode)

        results += beams

    return results


class HashTransformer:
    def __init__(
        self,
        n=5,
        num_range=4,
        query_format: PromptFormat = None,
        doc_format: PromptFormat = None,
        num_beams_doc=1,
        num_beams_query=1,
    ):
        self.n = n
        self.num_range = num_range
        self.query_format = query_format
        self.doc_format = doc_format
        self.num_beams_doc = num_beams_doc
        self.num_beams_query = num_beams_query

    def index(self, prompts, num_beams: int) -> tuple[np.array, list[str]]:
        pred_tokens_padded = text_to_int_list(
            prompts,
            n=self.n,
            num_range=self.num_range,
            num_beams=num_beams,
        )
        decoded_tokens_list = []
        for pred_tokens in pred_tokens_padded:
            decoded_tokens = "".join([str(token) for token in pred_tokens])
            decoded_tokens_list.append(decoded_tokens)

        return np.array(pred_tokens_padded).astype(int), decoded_tokens_list

    def index_doc(self, docs):
        if self.doc_format is not None:
            docs = self.doc_format.format(docs)

        return self.index(docs, self.num_beams_doc)

    def index_query(self, queries):
        if self.query_format is not None:
            queries = self.query_format.format(queries)

        return self.index(queries, self.num_beams_query)


def _test_prefix_tree_store():
    from termcolor import colored

    datas = get_sample_datasets(query_type="one2one")
    corpus_data = datas["corpus"]

    # Test using a simple hash transformer
    def test_insert_and_query(
        n,
        num_range,
        insertion_depth,
        msg,
        num_beams_doc=1,
        num_beams_query=1,
    ):
        specs = f"n={n}, num_range={num_range}, insertion_depth={insertion_depth}"
        print(colored("#" * 100, "grey"))
        print(colored(msg, "green"))
        print(colored(specs, "blue"))
        print(colored("#" * 100, "grey"))
        print()

        fbsearch_transformer = HashTransformer(
            n=n,
            num_range=num_range,
            num_beams_doc=num_beams_doc,
            num_beams_query=num_beams_query,
        )

        # Test on the sample dataset
        prefix_store = PrefixTreeStore(
            fbsearch_transformer,
            insertion_depth=insertion_depth,
        )
        prefix_store.insert(corpus_data)
        if DEBUG:
            print()
            print("Data store after inserting the corpus:")
            print(prefix_store.data_store)
            print()

        print("Testing on identical queries as the corpus plus a random query:")
        queries = corpus_data["content"][:3] + ["wrong query with no match"]
        doc_ids = [[doc_id] for doc_id in corpus_data["doc_id"][:3]] + [[]]
        results, metrics = prefix_store.eval_with_query(
            queries,
            doc_ids,
        )
        print("True:", doc_ids)
        print("Pred:")
        for result in results:
            print(result)
        print(colored(f"Stat: {metrics}", "red"))
        print()

    test_insert_and_query(
        n=5,
        num_range=2,
        insertion_depth=5,
        msg="SIMULATE THE CASE WHERE THE GENERATED TOKENS ARE **NOT** DIVERSE",
        num_beams_doc=1,
        num_beams_query=1,
    )

    test_insert_and_query(
        n=5,
        num_range=10,
        insertion_depth=5,
        msg="SIMULATE THE CASE WHERE THE GENERATED TOKENS ARE **YES** DIVERSE",
        num_beams_doc=1,
        num_beams_query=1,
    )

    test_insert_and_query(
        n=5,
        num_range=10,
        insertion_depth=1,
        msg="SIMULATE THE CASE WHERE THE GENERATED TOKENS ARE **YES** DIVERSE BUT THE INSERTION DEPTH IS SHALLOW",
        num_beams_doc=1,
        num_beams_query=1,
    )

    test_insert_and_query(
        n=5,
        num_range=10,
        insertion_depth=5,
        msg="SIMULATE THE CASE WHERE THE GENERATED TOKENS ARE **YES** DIVERSE",
        num_beams_doc=5,
        num_beams_query=5,
    )


if __name__ == "__main__":
    print("Running `python -m fbsearch.store.retrieval`")
    _test_prefix_tree_store()

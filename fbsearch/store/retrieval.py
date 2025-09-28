import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..model.llama import FBSearchTransformer
from ..model.prompt import PromptFormat

DEBUG = os.environ.get("DEBUG", False)


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
        content = self.corpus["content"][i]
        return {
            "doc_id": doc_id,
            "content": content,
        }


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
            "citations": citations,
            "query": query,
        }


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


def get_sample_datasets(file_dir=None, query_type="many2many"):
    if file_dir is None:
        file_dir = os.path.dirname(__file__)
    corpus_path = os.path.join(file_dir, "examples/retrieval/corpus.jsonl")
    query_path = os.path.join(file_dir, f"examples/retrieval/{query_type}.jsonl")

    corpus = pd.read_json(corpus_path, lines=True)
    queries = pd.read_json(query_path, lines=True)

    corpus_dataset = CorpusDataset(corpus)
    query2doc_dataset = Query2DocumentDataset(queries, corpus)
    return corpus_dataset, query2doc_dataset


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

    def insert(self, docs):
        doc_ids: list[int] = docs["doc_id"]
        contents: list[str] = docs["content"]
        genxs, tokens = self.transformer.index_doc(contents)
        if isinstance(genxs, torch.Tensor):
            genxs = genxs.cpu().numpy().astype(int)  # 2D numpy array

        for i in range(len(doc_ids)):
            doc_id = doc_ids[i]
            content = contents[i]
            genx = genxs[i]
            token = tokens[i]

            # Put into the data store
            self.data_store[doc_id] = {"content": content, "genx": genx, "token": token}

            # Insert into the prefix tree
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

    def query(self, queries: list[str]):
        genxs, tokens = self.transformer.index_query(queries)
        if isinstance(genxs, torch.Tensor):
            genxs = genxs.cpu().numpy().astype(int)  # 2D numpy array
        results = []
        for i in range(len(queries)):
            genx = genxs[i]
            result = self.traverse_and_search(genx)
            results.append(result)
        return results

    def eval_with_query(self, queries: list[str], doc_ids: list[list[int]]):
        results = self.query(queries)
        metrics = {}

        total_pred = 0
        total_pred_correct = 0
        total_labels = 0

        for idx in range(len(results)):
            result = results[idx]
            labels = doc_ids[idx]

            doc_ids_pred: list[int] = result["doc_ids"]
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
            return {"depth": depth, "doc_ids": node.doc_ids}
        return {"depth": -1, "doc_ids": set({})}


def text_to_int_list(texts, n=5, num_range=10):
    if not isinstance(texts, list):
        texts = [texts]

    results = []
    for text in texts:
        # Hash the input text using SHA256
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Convert the first n bytes to integers in the desired range
        hashcode = [digest[i] % num_range for i in range(n)]
        results.append(hashcode)

    return results


class HashTransformer:
    def __init__(
        self,
        n=5,
        num_range=4,
        query_format: PromptFormat = None,
        doc_format: PromptFormat = None,
    ):
        self.n = n
        self.num_range = num_range
        self.query_format = query_format
        self.doc_format = doc_format

    def index(self, prompts) -> tuple[np.array, list[str]]:
        pred_tokens_padded = text_to_int_list(
            prompts, n=self.n, num_range=self.num_range
        )
        decoded_tokens_list = []
        for pred_tokens in pred_tokens_padded:
            decoded_tokens = "".join([str(token) for token in pred_tokens])
            decoded_tokens_list.append(decoded_tokens)

        return np.array(pred_tokens_padded).astype(int), decoded_tokens_list

    def index_doc(self, docs):
        if self.doc_format is not None:
            docs = self.doc_format.format(docs)

        return self.index(docs)

    def index_query(self, queries):
        if self.query_format is not None:
            queries = self.query_format.format(queries)
        return self.index(queries)


if __name__ == "__main__":
    print("Running `python -m fbsearch.store.retrieval`")
    corpus_dataset, query2doc_dataset = get_sample_datasets(query_type="many2many")
    corpus_data = {
        "doc_id": [corpus_dataset[idx]["doc_id"] for idx in range(len(corpus_dataset))],
        "content": [
            corpus_dataset[idx]["content"] for idx in range(len(corpus_dataset))
        ],
    }
    print("The first 3 elements in corpus sample data:")
    print("IDs:", corpus_data["doc_id"][:3])
    print("Contents:", corpus_data["content"][:3])

    query2doc_data = {
        "doc_id": sum(
            [query2doc_dataset[idx]["doc_id"] for idx in range(len(query2doc_dataset))],
            [],
        ),
        "query": sum(
            [query2doc_dataset[idx]["query"] for idx in range(len(query2doc_dataset))],
            [],
        ),
        "content": sum(
            [
                query2doc_dataset[idx]["content"]
                for idx in range(len(query2doc_dataset))
            ],
            [],
        ),
    }
    print()
    print("Th first 2 elements in quer2doc sample data:")
    print("IDs:", query2doc_data["doc_id"][:2])
    print("Queries:", query2doc_data["query"][:2])
    print("Contents:", query2doc_data["content"][:2])
    print()

    # Test using a simple hash transformer
    def test_insert_and_query(n, num_range, insertion_depth, msg):
        specs = f"n={n}, num_range={num_range}, insertion_depth={insertion_depth}"
        print("#" * 100)
        print(msg)
        print(specs)
        print("#" * 100)
        print()

        fbsearch_transformer = HashTransformer(n=n, num_range=num_range)

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
        print(doc_ids)
        print(results)
        print(metrics)
        print()

    test_insert_and_query(
        n=5,
        num_range=2,
        insertion_depth=5,
        msg="SIMULATE THE CASE WHERE THE GENERATED TOKENS ARE **NOT** DIVERSE",
    )

    test_insert_and_query(
        n=5,
        num_range=10,
        insertion_depth=5,
        msg="SIMULATE THE CASE WHERE THE GENERATED TOKENS ARE **YES** DIVERSE",
    )

    test_insert_and_query(
        n=5,
        num_range=10,
        insertion_depth=1,
        msg="SIMULATE THE CASE WHERE THE GENERATED TOKENS ARE **YES** DIVERSE BUT THE INSERTION DEPTH IS SHALLOW",
    )

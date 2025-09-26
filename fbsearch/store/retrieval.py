import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..model.llama import FBSearchTransformer


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


def get_sample_query2doc_dataset():
    file_dir = os.path.dirname(__file__)
    corpus_path = os.path.join(file_dir, "examples/retrieval/corpus.jsonl")
    query_path = os.path.join(file_dir, "examples/retrieval/queries.jsonl")

    corpus = pd.read_json(corpus_path, lines=True)
    queries = pd.read_json(query_path, lines=True)

    query2doc_dataset = Query2DocumentDataset(queries, corpus)
    return query2doc_dataset


class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.doc_id = set()


class PrefixTreeStore:
    def __init__(
        self,
        transformer: FBSearchTransformer = None,
        insertion_depth: int = 5,
        columns: tuple[str, str] = ("doc_id", "content"),
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
        genxs = genxs.cpu().numpy().astype(int)  # 2D numpy array

        for i in range(len(doc_ids)):
            doc_id = doc_ids[i]
            content = contents[i]
            genx = genxs[i]

            # Put into the data store
            self.data_store[doc_id] = {"content": content, "genx": genx}

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
        for i in range(len(queries)):
            genx = genxs[i]
            results = self.traverse_and_search(genx)
            yield results

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
        return None


if __name__ == "__main__":
    print("Running `python -m fbsearch.dataset.retrieval`")
    query2doc_dataset = get_sample_query2doc_dataset()
    print(query2doc_dataset[0])

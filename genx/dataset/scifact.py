import pandas as pd
from torch.utils.data import DataLoader, Dataset


def row2text_template_scifact(row):
    return f"Title: {row['title']}\nAbstract: {' '.join(row['abstract'])}\nStructured: {row['structured']}\n"


class SciFactCorpusDataset(Dataset):
    def __init__(self, corpus_path):
        super().__init__()
        corpus = pd.read_json(corpus_path, lines=True)
        corpus["text"] = corpus.apply(row2text_template_scifact, axis=1)

        self.corpus = corpus
        self.corpus_dict = corpus.set_index("doc_id")["text"].to_dict()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, i):
        doc_id = self.corpus["doc_id"][i]
        text = self.corpus_dict.get(doc_id)
        return {
            "doc_id": doc_id,
            "text": text,
        }

    def get_corpus_dict(self):
        return self.corpus_dict


def get_scifact_corpus_dataloader(
    corpus_path,
    batch_size=4,
    shuffle=False,
    num_workers=0,
):
    scifact_corpus_dataset = SciFactCorpusDataset(corpus_path)
    dataloader = DataLoader(
        scifact_corpus_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


class SciFactQueryDataset(Dataset):
    def __init__(self, queries_path, corpus_dict):
        super().__init__()
        if isinstance(queries_path, str):
            self.queries = pd.read_json(queries_path, lines=True)
        elif isinstance(queries_path, list):
            queries = []
            for path in queries_path:
                queries.append(pd.read_json(path, lines=True))
            self.queries = pd.concat(queries, ignore_index=True)
        else:
            raise ValueError("queries_path must be a string or a list of strings")

        self.corpus_dict = corpus_dict

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        doc_id_list = self.queries["cited_doc_ids"][i]
        query = self.queries["claim"][i]

        docs = []
        for doc_id in doc_id_list:
            docs.append(self.corpus_dict.get(doc_id))

        n_docs = len(docs)

        if n_docs == 0:
            return {}
        else:
            queries = [query] * n_docs

        return {
            "doc_id": doc_id_list,
            "query": queries,
            "text": docs,
        }


def scifact_query_collate_fn(batch):
    batch = [item for item in batch if item]
    if not batch:
        return {}
    doc_ids = sum([item["doc_id"] for item in batch], [])
    queries = sum([item["query"] for item in batch], [])
    texts = sum([item["text"] for item in batch], [])
    return {
        "doc_id": doc_ids,
        "query": queries,
        "text": texts,
    }


def get_scifact_query_dataloader(
    queries_path: str | list[str],
    corpus_dict,
    batch_size=4,
    shuffle=False,
    num_workers=0,
):
    scifact_query_dataset = SciFactQueryDataset(
        queries_path,
        corpus_dict,
    )
    dataloader = DataLoader(
        scifact_query_dataset,
        batch_size=batch_size,
        collate_fn=scifact_query_collate_fn,  # NOTE: This is for query dataset only
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader

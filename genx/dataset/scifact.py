import pandas as pd
from torch.utils.data import DataLoader, Dataset


def row2text_template_scifact(row):
    return f"Title: {row['title']}\nAbstract: {' '.join(row['abstract'])}\nStructured: {row['structured']}\n"


class SciFactDataset(Dataset):
    def __init__(self, queries_path, corpus_path):
        super().__init__()
        queries = pd.read_json(queries_path, lines=True)
        corpus = pd.read_json(corpus_path, lines=True)
        corpus["text"] = corpus.apply(row2text_template_scifact, axis=1)

        self.queries = queries
        self.corpus = corpus
        self.corpus_dict = corpus.set_index("doc_id")["text"].to_dict()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        doc_id_list = self.queries["cited_doc_ids"][i]
        query = self.queries["claim"][i]

        docs = [
            self.corpus_dict.get(doc_id)
            for doc_id in doc_id_list
            if doc_id in self.corpus_dict
        ]
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


def scifact_collate_fn(batch):
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


def get_scifact_dataloader(
    queries_path,
    corpus_path,
    batch_size=4,
    shuffle=False,
    num_workers=0,
):
    dataset = SciFactDataset(queries_path, corpus_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=scifact_collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader

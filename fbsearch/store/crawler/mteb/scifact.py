# @noautodeps
"""
NOTE: This is a temporary draft for processing the scifact dataset.
"""

import pandas as pd

DATA_PATH = "/data/users/zy45/fbsource/fbcode/gen_ai/web_search/fbsearch/scripts/db/mteb/scifact/"
PROCESSED_PATH = DATA_PATH + "processed/"


def row2text_template_scifact(row):
    text = f"{row['title']} {' '.join(row['abstract'])}"
    # Remove all newlines and strip extra spaces
    text = text.replace("\n", " ")
    return text


corpus = pd.read_json(DATA_PATH + "corpus.jsonl", lines=True)
corpus["content"] = corpus.apply(row2text_template_scifact, axis=1)
# Save to jsonl file, only keep the doc_id column and content column
corpus_filtered = corpus[["doc_id", "content"]]
# Save to JSONL file
corpus_filtered.to_json(PROCESSED_PATH + "corpus.jsonl", orient="records", lines=True)
for flag in ["train", "dev"]:
    queries = pd.read_json(DATA_PATH + f"claims_{flag}.jsonl", lines=True).rename(
        columns={"claim": "query", "cited_doc_ids": "citations"}
    )
    queries_filtered = queries[["query", "citations"]]
    queries_filtered.to_json(
        PROCESSED_PATH + f"query_{flag}.jsonl", orient="records", lines=True
    )

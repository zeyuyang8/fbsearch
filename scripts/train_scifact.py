from dataclasses import dataclass, field

from fbsearch.model.trainer import (
    FBSearchTrainer,
    FBSearchTrainingArguments,
    get_fbsearch_transformer,
    HfArgumentParser,
)

from fbsearch.store.retrieval import get_retrieval_datasets


@dataclass
class DataArguments:
    corpus_path: str = field(
        default="/data/users/zy45/fbsource/fbcode/gen_ai/web_search/fbsearch/scripts/db/mteb/scifact/processed/corpus.jsonl"
    )
    query_train_path: str = field(
        default="/data/users/zy45/fbsource/fbcode/gen_ai/web_search/fbsearch/scripts/db/mteb/scifact/processed/query_train.jsonl"
    )
    query_dev_path: str = field(
        default="/data/users/zy45/fbsource/fbcode/gen_ai/web_search/fbsearch/scripts/db/mteb/scifact/processed/query_dev.jsonl"
    )


if __name__ == "__main__":
    parser = HfArgumentParser((FBSearchTrainingArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()

    datas = get_retrieval_datasets(
        corpus_path=data_args.corpus_path,
        query_train_path=data_args.query_train_path,
        query_dev_path=data_args.query_dev_path,
    )
    corpus_dataset = datas["corpus_dataset"]
    corpus_collate_fn = datas["corpus_collate_fn"]

    query2doc_dataset = datas["query2doc_dataset"]
    query2doc_collate_fn = datas["query2doc_collate_fn"]

    query_train_dataset = datas["query_train_dataset"]
    query_dev_dataset = datas["query_dev_dataset"]
    query_collate_fn = datas["query_collate_fn"]

    transformer = get_fbsearch_transformer(
        doc_model_name_or_path=args.doc_model_name_or_path,
        query_model_name_or_path=args.query_model_name_or_path,
        model_max_length=args.model_max_length,
        doc_prompt_before=args.doc_prompt_before,
        doc_prompt_after=args.doc_prompt_after,
        query_prompt_before=args.query_prompt_before,
        query_prompt_after=args.query_prompt_after,
        caching=args.caching,
        num_next_tokens=args.num_next_tokens,
        num_beams_doc=args.num_beams_doc,
        num_beams_query=args.num_beams_query,
    )

    trainer = FBSearchTrainer(
        transformer=transformer,
        args=args,
        # Corpus dataset
        corpus_dataset=corpus_dataset,
        # Query2doc dataset for training
        query2doc_dataset=query2doc_dataset,
        # Query dataset for evaluation
        query_train_dataset=query_train_dataset,
        query_dev_dataset=query_dev_dataset,
        # Collate functions
        corpus_collate_fn=corpus_collate_fn,
        query2doc_collate_fn=query2doc_collate_fn,
        query_collate_fn=query_collate_fn,
    )
    trainer.train()

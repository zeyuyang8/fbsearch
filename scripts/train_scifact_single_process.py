import copy
import itertools
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Otherwise running DDP will raise an error
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

TRANSFORMERS_PATH_MAP = {
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

############################################ MD ############################################


def get_special_tokens_dict(tokenizer):
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return special_tokens_dict


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_model_and_tokenizer(
    model_name_or_path,
    model_max_length=1024,
    torch_dtype=torch.bfloat16,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="left",
        use_fast=False,
    )
    special_tokens_dict = get_special_tokens_dict(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    return model, tokenizer


def get_genx_transformer(
    query_model_name_or_path,
    doc_model_name_or_path,
    query_model_max_length=128,
    doc_model_max_length=512,
    num_beams: int = 5,
    num_tokens: int = 5,
    torch_dtype=torch.bfloat16,
):
    query_model, query_tokenizer = get_model_and_tokenizer(
        query_model_name_or_path,
        model_max_length=query_model_max_length,
        torch_dtype=torch_dtype,
    )
    doc_model, doc_tokenizer = get_model_and_tokenizer(
        doc_model_name_or_path,
        model_max_length=doc_model_max_length,
        torch_dtype=torch_dtype,
    )
    return GenXTransformer(
        query_model=query_model,
        doc_model=doc_model,
        query_tokenizer=query_tokenizer,
        doc_tokenizer=doc_tokenizer,
        num_beams=num_beams,
        num_tokens=num_tokens,
    )


class GenXTransformer(nn.Module):
    def __init__(
        self,
        query_model,
        doc_model,
        query_tokenizer,
        doc_tokenizer,
        num_beams: int = 5,
        num_tokens: int = 5,
    ):
        super().__init__()
        self.query_model = query_model
        self.doc_model = doc_model

        self.query_model.train()
        self.doc_model.eval()

        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

        self.num_beams = num_beams
        self.num_tokens = num_tokens

        self.verbose = False

        self.config_genx_gen_kwargs(num_beams=num_beams, num_tokens=num_tokens)

    def set_train_eval_mode(self, query_train: bool = True, doc_train: bool = False):
        if query_train:
            self.query_model.train()
        else:
            self.query_model.eval()
        if doc_train:
            self.doc_model.train()
        else:
            self.doc_model.eval()

    def config_genx_gen_kwargs(self, **kwargs):
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 5),
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 30),
            "top_p": kwargs.get("top_p", 0.95),
            "do_sample": False,
            "num_beams": kwargs.get("num_beams", 5),
            "num_return_sequences": kwargs.get("num_return_sequences", 5),
            "eos_token_id": kwargs.get("eos_token_id", None),
            "pad_token_id": kwargs.get("pad_token_id", None),
        }
        self.genx_gen_kwargs = gen_kwargs

    def sample_beams_of_next_tokens(
        self,
        model,
        tokenizer,
        prompts: list[str],
    ) -> list[list[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        breakpoint()
        device = model.device

        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
        )
        batch["input_len"] = len(batch["input_ids"][0])

        gen_kwargs = self.genx_gen_kwargs.copy()

        with torch.no_grad():
            gen_kwargs["input_ids"] = batch["input_ids"].to(device)
            gen_kwargs["attention_mask"] = batch["attention_mask"].to(device)
            generated_tokens = model.generate(**gen_kwargs)

        input_len = batch["input_len"]
        pred_next_tokens = generated_tokens[:, input_len:]
        if self.verbose:
            print(
                "Decoded tokens:",
                tokenizer.batch_decode(pred_next_tokens, skip_special_tokens=False),
            )

        batch_size = len(prompts)
        num_return_sequences = gen_kwargs["num_return_sequences"]

        pred_next_tokens = pred_next_tokens.view(batch_size, num_return_sequences, -1)
        pred_next_tokens = pred_next_tokens.cpu().tolist()

        print("Token IDs:", pred_next_tokens) if self.verbose else None
        return pred_next_tokens

    def get_sft_loss_txt(self, model, tokenizer, prompts: list[str]):
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        labels = input_ids.clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        return loss

    def forward(self, queries: list[str], docs: list[str]):
        # Now this is fine-tuning query model to generate next tokens of document
        assert len(queries) == len(docs)

        beams_for_docs: list[list[str]] = self.sample_beams_of_next_tokens(
            self.doc_model,
            self.doc_tokenizer,
            docs,
        )  # Shape is num_docs x num_beams x num_tokens

        # Shape is num_docs x num_beams x (len(query) + num_tokens)
        prompts_for_all_pairs: list[list[str]] = []
        for doc_idx, beams in enumerate(beams_for_docs):
            beams = self.doc_tokenizer.batch_decode(beams, skip_special_tokens=True)
            prompts = []  # List of the same query and num_beams possible next sentences

            query = queries[doc_idx]
            num_beams = len(beams)
            for beams_idx in range(num_beams):
                prompt = query + beams[beams_idx]
                prompts.append(prompt)

            prompts_for_all_pairs.append(prompts)

        # Have num_docs x num_beams sequences, each of a string of length (len(query) + num_tokens)
        flats: list[str] = list(itertools.chain.from_iterable(prompts_for_all_pairs))

        loss = self.get_sft_loss_txt(self.query_model, self.query_tokenizer, flats)
        return loss


############################################ DT ############################################


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


def get_scifact_dataloader(queries_path, corpus_path, batch_size=4, shuffle=False):
    dataset = SciFactDataset(queries_path, corpus_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=scifact_collate_fn,
        shuffle=shuffle,
    )
    return dataloader


def compare_tokenizer(tokenizer1, tokenizer2):
    # Quick identity check
    if tokenizer1 is tokenizer2:
        return True

    # Check basic properties
    if tokenizer1.vocab_size != tokenizer2.vocab_size:
        return False

    # More comprehensive test strings
    test_strings = [
        "Hello world!",
        "Meta AI is awesome.",
        "Tokenization test: 12345.",
        "",  # Empty string
        " ",  # Whitespace only
        "🤗 Unicode: café naïve",  # Unicode/emoji
        "A" * 100,  # Long string
    ]

    # Compare both tokenization and encoding
    for s in test_strings:
        try:
            # Compare token strings
            tokens1 = tokenizer1.tokenize(s)
            tokens2 = tokenizer2.tokenize(s)
            if tokens1 != tokens2:
                return False

            # Compare token IDs (more important)
            encoded1 = tokenizer1.encode(s, add_special_tokens=False)
            encoded2 = tokenizer2.encode(s, add_special_tokens=False)
            if encoded1 != encoded2:
                return False

        except Exception:
            # If tokenization behaves differently (one fails, other doesn't)
            return False

    # Check special tokens
    special_tokens = ["pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"]
    for attr in special_tokens:
        if getattr(tokenizer1, attr, None) != getattr(tokenizer2, attr, None):
            return False

    return True


@dataclass
class Arguments:
    # Model
    query_model_alias: str = field(default="llama-3.2-1b-instruct")
    doc_model_alias: str = field(default="llama-3.2-1b-instruct")
    query_model_max_length: int = field(default=128)
    doc_model_max_length: int = field(default=1024)
    torch_dtype: str = field(default="bfloat16")

    # Indexing
    num_next_tokens: int = field(default=5)

    # Prompts
    doc_prompt_before: str = field(default="")

    # Data
    data_path: str = field(default="/home/zy45/code/genx/scripts/data/scifact")
    train_queries_filename: str = field(default="claims_train.jsonl")
    dev_queries_filename: str = field(default="claims_dev.jsonl")
    syn_queries_filename: str = field(default="claims_syn_all_dedup.jsonl")
    corpus_filename: str = field(default="corpus.jsonl")

    # Device
    device: str = field(default="cuda")


############################################ SP ############################################


def single_process(args, training_args):
    # Arguments
    torch_dtype = getattr(torch, args.torch_dtype)

    # Data
    data_path = args.data_path
    train_queries_path = os.path.join(data_path, args.train_queries_filename)
    dev_queries_path = os.path.join(data_path, args.dev_queries_filename)
    syn_queries_path = os.path.join(data_path, args.syn_queries_filename)
    corpus_path = os.path.join(data_path, args.corpus_filename)

    batch_size = training_args.per_device_train_batch_size
    train_dataloader = get_scifact_dataloader(
        queries_path=train_queries_path,
        corpus_path=corpus_path,
        batch_size=batch_size,
    )
    dev_dataloader = get_scifact_dataloader(
        queries_path=dev_queries_path,
        corpus_path=corpus_path,
        batch_size=batch_size,
    )
    syn_dataloader = get_scifact_dataloader(
        queries_path=syn_queries_path,
        corpus_path=corpus_path,
        batch_size=batch_size,
    )

    # Model
    print("Loading models...")
    query_model_name_or_path = TRANSFORMERS_PATH_MAP[args.query_model_alias]
    doc_model_name_or_path = TRANSFORMERS_PATH_MAP[args.doc_model_alias]
    genx_transformer: GenXTransformer = get_genx_transformer(
        query_model_name_or_path=query_model_name_or_path,
        doc_model_name_or_path=doc_model_name_or_path,
        query_model_max_length=args.query_model_max_length,
        doc_model_max_length=args.doc_model_max_length,
    )
    genx_transformer.to("cuda")

    # Simple test
    batch = next(iter(train_dataloader))
    queries = batch["query"]
    docs = batch["text"]
    print("Queries:", queries)
    print("Docs:", docs)
    print("SFT loss:", genx_transformer(queries, docs))

    # Training
    num_train_epochs = training_args.num_train_epochs
    learning_rate = training_args.learning_rate
    print(f"Number of training epochs: {num_train_epochs}")
    print(f"Learning rate: {learning_rate}")


############################################ HF ############################################


class GenXTrainer(Trainer):
    def __init__(
        self,
        model: GenXTransformer,
        args,
        train_dataset,
        data_collator,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    def get_train_dataloader(self):
        train_dataset: Dataset = self.train_dataset
        sampler = self._get_train_sampler()  # This ensures DDP compatibility
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        queries = inputs["query"]
        docs = inputs["text"]

        loss = model(queries, docs)
        print("SFT loss:", loss)
        raise NotImplementedError
        outputs = ...
        return (loss, outputs) if return_outputs else loss


def ddp_process(args, training_args):
    print("DDP training")

    # Arguments
    torch_dtype = getattr(torch, args.torch_dtype)

    # Data
    data_path = args.data_path
    train_queries_path = os.path.join(data_path, args.train_queries_filename)
    dev_queries_path = os.path.join(data_path, args.dev_queries_filename)
    syn_queries_path = os.path.join(data_path, args.syn_queries_filename)
    corpus_path = os.path.join(data_path, args.corpus_filename)

    train_dataset = SciFactDataset(
        queries_path=train_queries_path,
        corpus_path=corpus_path,
    )

    # Model
    print("Loading models...")
    query_model_name_or_path = TRANSFORMERS_PATH_MAP[args.query_model_alias]
    doc_model_name_or_path = TRANSFORMERS_PATH_MAP[args.doc_model_alias]
    genx_transformer: GenXTransformer = get_genx_transformer(
        query_model_name_or_path=query_model_name_or_path,
        doc_model_name_or_path=doc_model_name_or_path,
        query_model_max_length=args.query_model_max_length,
        doc_model_max_length=args.doc_model_max_length,
    )

    trainer = GenXTrainer(
        model=genx_transformer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=scifact_collate_fn,
    )
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    # single_process(args, training_args)
    ddp_process(args, training_args)

import os

import torch
import transformers.utils.logging as transformers_utils_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .prompt import PromptFormat

DEBUG = os.environ.get("DEBUG", False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    transformers_utils_logging.set_verbosity_error()
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


def get_special_tokens_dict(tokenizer):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return special_tokens_dict


def get_tokenizer(model_name_or_path, model_max_length, mode):
    if mode == "train":
        padding_side = "right"
    elif mode == "eval":
        padding_side = "left"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side=padding_side,
        use_fast=False,
    )
    return tokenizer


def get_llm_and_tokenizer(
    model_name_or_path,
    mode: str,
    model_max_length: int = 2048,
    dtype=torch.bfloat16,
):
    tokenizer = get_tokenizer(model_name_or_path, model_max_length, mode)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        dtype=dtype,
    )
    special_tokens_dict = get_special_tokens_dict(tokenizer)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    return model, tokenizer


def batch_insert_tensors(mask1, mask2, ids1, ids2):
    IGNORE_INDEX = -100
    batch_size = mask1.shape[0]
    seq_len2 = mask2.shape[1]
    mask_length = mask1.shape[1] + seq_len2

    # Create mask tensor
    mask_all = torch.empty(
        batch_size, mask_length, dtype=mask1.dtype, device=mask1.device
    )
    ids_all = torch.empty_like(mask_all)
    labels_all = torch.empty_like(mask_all)

    # Process each row
    for i in range(batch_size):
        # Find split point for this row
        split_idx = (mask1[i] != 0).sum().item()

        # Copy first part
        mask_all[i, :split_idx] = mask1[i, :split_idx]
        ids_all[i, :split_idx] = ids1[i, :split_idx]
        labels_all[i, :split_idx] = IGNORE_INDEX

        # Insert
        mask_all[i, split_idx : split_idx + seq_len2] = mask2[i]
        ids_all[i, split_idx : split_idx + seq_len2] = ids2[i]
        labels_all[i, split_idx : split_idx + seq_len2] = ids2[i]

        # Copy remaining part
        mask_all[i, split_idx + seq_len2 :] = mask1[i, split_idx:]
        ids_all[i, split_idx + seq_len2 :] = ids1[i, split_idx:]
        labels_all[i, split_idx + seq_len2 :] = IGNORE_INDEX

    return mask_all, ids_all, labels_all


class FBSearchTransformer:
    def __init__(
        self,
        query_model: AutoModelForCausalLM = None,
        doc_model: AutoModelForCausalLM = None,
        train_tokenizer: PreTrainedTokenizer = None,
        eval_tokenizer: PreTrainedTokenizer = None,
        query_format: PromptFormat = None,
        doc_format: PromptFormat = None,
        caching: bool = True,
    ):
        self.query_model = query_model
        self.doc_model = doc_model

        self.train_tokenizer = train_tokenizer
        self.eval_tokenizer = eval_tokenizer

        self.query_format = query_format
        self.doc_format = doc_format

        if caching:
            self.doc_cache = {}

    def to(self, device):
        self.query_model.to(device)
        self.doc_model.to(device)
        return self

    def index(
        self,
        prompts: str,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        mode: str = "query",
    ) -> tuple[torch.Tensor, list[str]]:
        assert tokenizer.padding_side == "left"

        model.eval()
        device = model.device
        if isinstance(prompts, str):
            prompts = [prompts]

        gen_kwargs = {
            "min_new_tokens": 5,
            "max_new_tokens": 5,
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
        }

        pred_tokens_list = []
        decoded_tokens_list = []
        for prompt in prompts:
            if (
                self.doc_cache is not None
                and mode == "doc"
                and prompt in self.doc_cache
            ):
                pred_tokens, decoded = self.doc_cache[prompt]
                pred_tokens = pred_tokens.to(device)
                pred_tokens_list.append(pred_tokens)
                decoded_tokens_list.append(decoded)
                continue

            batch = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,  # No padding needed for single prompt
            )
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            input_len = input_ids.shape[1]
            with torch.no_grad():
                if hasattr(model, "module") and isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ):
                    gen_model = model.module
                else:
                    gen_model = model

                generated_tokens = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    **gen_kwargs,
                )
            pred_tokens = generated_tokens[:, input_len:][0]  # Remove batch dim
            pred_tokens_list.append(pred_tokens)
            decoded = tokenizer.decode(
                pred_tokens,
                skip_special_tokens=True,
            )
            decoded_tokens_list.append(decoded)

            if self.doc_cache is not None and mode == "doc":
                # Make sure the doc model is freezed, otherwise we should not cache it
                all_frozen = all(
                    not param.requires_grad for param in self.doc_model.parameters()
                )
                assert all_frozen, "Model must be frozen for caching"
                self.doc_cache[prompt] = (pred_tokens.cpu(), decoded)

        # Pad pred_tokens_list to the same length (they are already having the same length)
        # this will return a 2D tensor with shape (batch_size, max_new_tokens)
        pred_tokens_padded = torch.nn.utils.rnn.pad_sequence(
            pred_tokens_list, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        return pred_tokens_padded, decoded_tokens_list

    def index_doc(self, docs: list[str]):
        assert self.eval_tokenizer.padding_side == "left"

        if self.doc_format is not None:
            docs = self.doc_format.format(docs)

        return self.index(docs, self.doc_model, self.eval_tokenizer, mode="doc")

    def index_query(self, queries: list[str]):
        assert self.eval_tokenizer.padding_side == "left"

        if self.query_format is not None:
            queries = self.query_format.format(queries)

        return self.index(queries, self.query_model, self.eval_tokenizer, mode="query")

    def compute_loss(self, queries: list[str], docs: list[str]):
        assert self.eval_tokenizer.padding_side == "left"
        assert self.train_tokenizer.padding_side == "right"

        device = self.query_model.device

        if isinstance(queries, str):
            queries = [queries]
        if isinstance(docs, str):
            docs = [docs]
        assert len(queries) == len(docs)

        # Docs are formatted in `self.index_doc`
        targets_ids, targets_decoded = self.index_doc(docs)
        targets_mask = torch.ones_like(targets_ids)
        if DEBUG:
            print(targets_decoded)
            print()
            print("Targets")
            print(targets_ids)
            print(targets_mask)
            print()

        # Remember to format queries before computing loss
        if self.query_format is not None:
            queries = self.query_format.format(queries)
        sources = self.train_tokenizer(
            queries,
            return_tensors="pt",
            padding="longest",
        ).to(device)
        sources_ids = sources["input_ids"]
        sources_mask = sources["attention_mask"]
        if DEBUG:
            print("Sources")
            print(sources_ids)
            print(sources_mask)
            print()

        sources_targets_mask, sources_targets_ids, sources_targets_labels = (
            batch_insert_tensors(sources_mask, targets_mask, sources_ids, targets_ids)
        )
        if DEBUG:
            print("Combined")
            print(sources_targets_ids)
            print(sources_targets_mask)
            print(sources_targets_labels)
            print()

        self.query_model.train()
        outputs = self.query_model(
            input_ids=sources_targets_ids,
            attention_mask=sources_targets_mask,
            labels=sources_targets_labels,
        )
        if DEBUG:
            print("Loss")
            print(outputs.loss)
        return outputs.loss


def get_fbsearch_transformer(
    doc_model_name_or_path: str,
    query_model_name_or_path: str,
    model_max_length: int = 2048,
    doc_prompt_before: str = "Document: ",
    doc_prompt_after: str = "Task: Generate a summary with several words. Directly say the words without explanation.",
    query_prompt_before: str = "Query: ",
    query_prompt_after: str = "Task: Guess you answer with several words. Directly say the words without explanation.",
    dtype: torch.dtype = torch.bfloat16,
):
    doc_model, eval_tokenizer = get_llm_and_tokenizer(
        doc_model_name_or_path,
        mode="eval",
        model_max_length=model_max_length,
    )
    query_model, train_tokenizer = get_llm_and_tokenizer(
        query_model_name_or_path,
        mode="train",
        model_max_length=model_max_length,
    )
    doc_format = PromptFormat(
        before=doc_prompt_before,
        after=doc_prompt_after,
    )
    query_format = PromptFormat(
        before=query_prompt_before,
        after=query_prompt_after,
    )
    fbsearch_transformer = FBSearchTransformer(
        query_model=query_model,
        doc_model=doc_model,
        train_tokenizer=train_tokenizer,
        eval_tokenizer=eval_tokenizer,
        query_format=query_format,
        doc_format=doc_format,
    )
    return fbsearch_transformer


if __name__ == "__main__":
    print("Running `python -m fbsearch.model.llama`")
    model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    model_max_length = 2048

    # FBSearch transformer
    fbsearch_transformer = get_fbsearch_transformer(
        doc_model_name_or_path=model_name_or_path,
        query_model_name_or_path=model_name_or_path,
        model_max_length=model_max_length,
    ).to("cuda")

    # Test indexing docs and queries
    docs = ["The capital of France is Paris.", "Micheal Jordan is the GOAT."]
    queries = ["What is capital of France?", "Who is the GOAT of basketball?"]

    print(fbsearch_transformer.index_doc(docs))
    print(fbsearch_transformer.index_query(queries))

    # Test computing loss
    loss = fbsearch_transformer.compute_loss(queries, docs)
    print(f"Loss: {loss}")

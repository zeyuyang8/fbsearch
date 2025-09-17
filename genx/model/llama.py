import itertools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_special_tokens_dict(tokenizer):
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
    num_next_tokens: int = 5,
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
        num_next_tokens=num_next_tokens,
    )


class GenXTransformer:
    def __init__(
        self,
        query_model,
        doc_model,
        query_tokenizer,
        doc_tokenizer,
        num_beams: int = 5,
        num_next_tokens: int = 5,
        save_cache: bool = False,
    ):
        super().__init__()
        self.query_model = query_model
        self.doc_model = doc_model

        # Use cache
        self.query_model.config.use_cache = True
        self.doc_model.config.use_cache = True

        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

        self.num_beams = num_beams
        self.num_next_tokens = num_next_tokens

        self.verbose = False

        self.config_genx_gen_kwargs(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            num_next_tokens=num_next_tokens,
        )

        self.save_cache = save_cache
        self.cache = {}

    def set_train_eval_mode(self, query_train: bool = True, doc_train: bool = False):
        if query_train:
            self.query_model.train()
        else:
            self.query_model.eval()
        if doc_train:
            self.doc_model.train()
        else:
            self.doc_model.eval()

    def update_num_next_tokens(self, num_next_tokens: int):
        self.num_next_tokens = num_next_tokens
        self.genx_gen_kwargs["max_new_tokens"] = num_next_tokens

    def config_genx_gen_kwargs(self, **kwargs):
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 5),
            "do_sample": False,
            "num_beams": kwargs.get("num_beams", 5),
            "num_return_sequences": kwargs.get("num_return_sequences", 5),
            "eos_token_id": kwargs.get("eos_token_id", None),
            "pad_token_id": kwargs.get("pad_token_id", None),
        }
        self.genx_gen_kwargs = gen_kwargs

    def index_prompt(self, prompts, model, tokenizer):
        device = model.device

        if isinstance(prompts, str):
            prompts = [prompts]

        batch = tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
        )
        batch["input_len"] = len(batch["input_ids"][0])

        genx_gen_kwargs = self.genx_gen_kwargs.copy()
        with torch.no_grad():
            genx_gen_kwargs["input_ids"] = batch["input_ids"].to(device)
            genx_gen_kwargs["attention_mask"] = batch["attention_mask"].to(device)
            generated_tokens = model.generate(**genx_gen_kwargs)

        input_len = batch["input_len"]
        pred_next_tokens = generated_tokens[:, input_len:]
        if self.verbose:
            print(
                "Decoded tokens:",
                tokenizer.batch_decode(pred_next_tokens, skip_special_tokens=False),
            )

        batch_size = len(prompts)
        num_return_sequences = genx_gen_kwargs["num_return_sequences"]

        pred_next_tokens = pred_next_tokens.view(batch_size, num_return_sequences, -1)
        pred_next_tokens = pred_next_tokens.cpu().tolist()

        print("Token IDs:", pred_next_tokens) if self.verbose else None
        return pred_next_tokens

    def index_query(self, prompts: list[str]):
        return self.index_prompt(prompts, self.query_model, self.query_tokenizer)

    def index_doc(self, prompts: list[str]):
        return self.index_prompt(prompts, self.doc_model, self.doc_tokenizer)

    def sample_beams_of_next_tokens(
        self,
        model,
        tokenizer,
        prompts: list[str],
    ) -> list[list[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]

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

        # Mask out all but last 5 tokens
        # shape: (batch_size, seq_len)
        batch_size, seq_len = input_ids.size()
        keep = self.num_next_tokens
        mask = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(
            batch_size, -1
        ) < (seq_len - keep)

        labels[mask] = -100  # -100 tells HF loss function to ignore those positions

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        return loss

    def __call__(self, queries: list[str], docs: list[str]):
        # Now this is fine-tuning query model to generate next tokens of document
        assert len(queries) == len(docs)

        beams_for_docs: list[list[list[int]]] = self.sample_beams_of_next_tokens(
            self.doc_model,
            self.doc_tokenizer,
            docs,
        )  # Shape is num_docs x num_beams x num_next_tokens

        # Shape is num_docs x num_beams x (len(query) + num_next_tokens)
        prompts_for_all_pairs: list[list[str]] = []
        for doc_idx, beams in enumerate(beams_for_docs):
            beams = self.doc_tokenizer.batch_decode(beams, skip_special_tokens=False)
            prompts = []  # List of the same query and num_beams possible next sentences

            query = queries[doc_idx]
            num_beams = len(beams)
            for beams_idx in range(num_beams):
                prompt = query + beams[beams_idx]
                prompts.append(prompt)

            prompts_for_all_pairs.append(prompts)

        # Have num_docs x num_beams sequences, each of a string of length (len(query) + num_next_tokens)
        flats: list[str] = list(itertools.chain.from_iterable(prompts_for_all_pairs))

        loss = self.get_sft_loss_txt(self.query_model, self.query_tokenizer, flats)
        return loss

import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.distributed as dist

from genx.utils import cleanup_ddp, setup_ddp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

# Constants
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class PrompterArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.2-3B-Instruct")
    model_max_length: int = field(default=512)
    batch_size: int = field(default=16)


def corpus_row_to_text_template_scifact(row):
    return f"Title: {row['title']}\nAbstract: {' '.join(row['abstract'])}\nStructured: {row['structured']}\n"


def generate_prompts_for_queries(paragraph, length="short", num_queries=20):
    question_prompt = f"\nBased on the text above, create {num_queries} diverse questions that vary in {length} and difficulty. Include simple factual questions as well as more complex ones that require reading between the lines or making connections. Make sure every question can be fully answered using only the information provided in the text. Respond with ONLY a numbered list with XML items in this exact format:\n\n1. <q>Question 1</q>\n2. <q>Question 2</q>\n3. <q>Question 3</q>\n...\n{num_queries}. <q>Question {num_queries}</q>\n\nSTOP. Do not add any additional text."
    question = f"{paragraph}{question_prompt}"
    return question


def extract_queries(sentence_pred):
    # Extract numbered questions (assuming format: "1. Question text")
    queries = re.findall(r"<q>(.*?)</q>", sentence_pred, re.DOTALL)
    queries = [query.strip() for query in queries]
    return queries


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


def split_batches_for_ddp(questions_data, rank, world_size):
    """Split batches across processes"""
    batches_per_process = len(questions_data) // world_size
    start_idx = rank * batches_per_process

    if rank == world_size - 1:  # Last process handles remainder
        end_idx = len(questions_data)
    else:
        end_idx = start_idx + batches_per_process

    return questions_data[start_idx:end_idx]


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


def main():
    """Main generation function"""
    # Setup DDP
    local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    print(f"Process {rank}/{world_size} running on device {device}")

    parser = HfArgumentParser(PrompterArguments)
    (prompter_args,) = parser.parse_args_into_dataclasses()

    # Load your model here
    prompter_tokenizer = AutoTokenizer.from_pretrained(
        prompter_args.model_name_or_path,
        model_max_length=prompter_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # Add special tokens to tokenizer
    special_tokens_dict = get_special_tokens_dict(prompter_tokenizer)

    prompter_model = AutoModelForCausalLM.from_pretrained(
        prompter_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    prompter_model = prompter_model.to(device)
    prompter_model.eval()

    # Add special tokens to model and resize embeddings
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=prompter_tokenizer,
        model=prompter_model,
    )
    prompter_model = DDP(prompter_model, device_ids=[local_rank])

    # Load your data here
    corpus = pd.read_json("./data/scifact/corpus.jsonl", lines=True)

    questions = {}
    for _, row in corpus.iterrows():
        corpus_id = row["doc_id"]
        corpus_text = corpus_row_to_text_template_scifact(row)
        question = generate_prompts_for_queries(corpus_text)
        # print(question)
        questions[corpus_id] = question

    # Extract the text values from the dictionary
    question_texts = list(questions.values())
    # Optionally keep track of doc_ids if needed
    doc_ids = list(questions.keys())

    batch_size = prompter_args.batch_size
    n_batches = len(question_texts) // batch_size
    questions_data = []

    for i in range(n_batches):
        batch_texts = question_texts[i * batch_size : (i + 1) * batch_size]

        batch = prompter_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="longest",
        )

        # These lines need to be inside the loop and properly indented
        batch["input_len"] = len(batch["input_ids"][0])
        batch["doc_ids"] = doc_ids[i * batch_size : (i + 1) * batch_size]

        questions_data.append(batch)

    gen_kwargs = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    # Set different seeds for each process to avoid identical outputs
    set_seed(42 + rank)

    # Split batches for this process
    local_batches = split_batches_for_ddp(questions_data, rank, world_size)
    local_records = []

    print(f"Process {rank} processing {len(local_batches)} batches")

    for step, batch in enumerate(local_batches):
        if step % 10 == 0:
            print(f"Process {rank}: batch {step}/{len(local_batches)}")

        with torch.no_grad():
            gen_kwargs["input_ids"] = batch["input_ids"].to(device)
            gen_kwargs["attention_mask"] = batch["attention_mask"].to(device)
            generated_tokens = prompter_model.module.generate(
                **gen_kwargs
            )  # Use .module for DDP

        pred_tokens = generated_tokens[:, batch["input_len"] :]
        decoded_pred = prompter_tokenizer.batch_decode(
            pred_tokens, skip_special_tokens=True
        )
        queries = [list(set(extract_queries(item))) for item in decoded_pred]
        doc_ids = batch["doc_ids"]

        for doc_id, query_list in zip(doc_ids, queries):
            local_records.append({"doc_id": doc_id, "query": query_list})

    # Save local results to temporary file
    temp_file = f"temp_data_rank_{rank}.jsonl"
    with open(temp_file, "w") as f:
        for record in local_records:
            json.dump(record, f)
            f.write("\n")

    print(f"Process {rank} finished, saved {len(local_records)} records to {temp_file}")

    # Synchronize all processes
    dist.barrier()

    # Only rank 0 merges all results
    if rank == 0:
        print("Merging results from all processes...")
        all_records = []

        # Read from all temporary files
        for r in range(world_size):
            temp_file = f"temp_data_rank_{r}.jsonl"
            if os.path.exists(temp_file):
                with open(temp_file, "r") as f:
                    for line in f:
                        all_records.append(json.loads(line))
                # Clean up temp file
                os.remove(temp_file)

        print(f"Total records collected: {len(all_records)}")

        # Remove duplicates and write final result
        seen_records = set()
        syn_queries_path = "./data/scifact/claims_syn_all.jsonl"
        syn_clean_queries_path = "./data/scifact/claims_syn_all_dedup.jsonl"

        with open(syn_queries_path, "w") as f:
            for record in all_records:
                record_key = (record["doc_id"], tuple(record["query"]))
                if record_key not in seen_records:
                    json.dump(record, f)
                    f.write("\n")
                    seen_records.add(record_key)

        print(
            f"Finished! Generated {len(all_records)} total records, {len(seen_records)} unique records."
        )

        syn_queries = pd.read_json(syn_queries_path, lines=True)
        items = []
        for _, row in syn_queries.iterrows():
            doc_id = row["doc_id"]
            queries = row["query"]

            for query in queries:
                items.append(
                    {
                        "claim": query,
                        "cited_doc_ids": [int(doc_id)],
                    }
                )

        df = pd.DataFrame(items)
        df_unique = df[~df["claim"].duplicated(keep=False)]
        df_unique.to_json(syn_clean_queries_path, orient="records", lines=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()

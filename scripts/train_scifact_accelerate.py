import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import itertools
import logging
import math
from dataclasses import dataclass, field

import torch
import transformers.utils.logging as transformers_utils_logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from genx.const import TRANSFORMERS_PATH_MAP
from genx.dataset.scifact import get_scifact_dataloader
from genx.model.llama import GenXTransformer, get_genx_transformer
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_scheduler, HfArgumentParser

logger = get_logger(__name__)


@dataclass
class Arguments:
    # Model
    query_model_alias: str = field(default="llama-3.2-1b-instruct")
    doc_model_alias: str = field(default="llama-3.2-1b-instruct")
    query_model_max_length: int = field(default=128)
    doc_model_max_length: int = field(default=2048)
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

    # Logging
    output_dir: str = field(default="./runs")
    logging_dir: str = field(default="logs")
    tracker_name: str = field(default="test")

    # Device
    device: str = field(default="cuda")

    # Seed
    seed: int = field(default=0)

    # Accelerator
    mixed_precision: str = field(default="bf16")
    report_to: str = field(default="wandb")
    do_report: bool = field(default=True)

    # Training
    per_device_train_batch_size: int = field(default=4)
    lr_warmup_steps: int = field(default=100)
    lr_scheduler: str = field(default="cosine")
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: int = field(default=10)
    learning_rate: float = field(default=1e-5)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    adam_weight_decay: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    dataloader_num_workers: int = field(default=4)
    validation_epochs: int = field(default=1)

    # Resume checkpoint
    resume_from_checkpoint: str = field(default=None)
    checkpointing_steps: int = field(default=1000)


def ddp_process(args):
    print("DDP training")

    # Accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers_utils_logging.set_verbosity_warning()
    else:
        transformers_utils_logging.set_verbosity_error()

    # Seed
    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Data
    data_path = args.data_path
    train_queries_path = os.path.join(data_path, args.train_queries_filename)
    corpus_path = os.path.join(data_path, args.corpus_filename)

    per_device_train_batch_size = args.per_device_train_batch_size
    train_dataloader = get_scifact_dataloader(
        queries_path=train_queries_path,
        corpus_path=corpus_path,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Data type
    torch_dtype = getattr(torch, args.torch_dtype)

    # Model
    print("Loading models...")
    query_model_name_or_path = TRANSFORMERS_PATH_MAP[args.query_model_alias]
    doc_model_name_or_path = TRANSFORMERS_PATH_MAP[args.doc_model_alias]
    genx_transformer: GenXTransformer = get_genx_transformer(
        query_model_name_or_path=query_model_name_or_path,
        doc_model_name_or_path=doc_model_name_or_path,
        query_model_max_length=args.query_model_max_length,
        doc_model_max_length=args.doc_model_max_length,
        torch_dtype=torch_dtype,
    )
    query_model, doc_model = genx_transformer.query_model, genx_transformer.doc_model
    query_model.to(accelerator.device, dtype=torch_dtype)
    doc_model.to(accelerator.device, dtype=torch_dtype)

    # Only train the query model
    query_model.train()
    doc_model.eval()
    doc_model.requires_grad_(False)

    # Optimizer and learning rate scheduler
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    len_train_dataloader_after_sharding = math.ceil(
        len(train_dataloader) / accelerator.num_processes
    )
    num_update_steps_per_epoch = math.ceil(
        len_train_dataloader_after_sharding / args.gradient_accumulation_steps
    )
    num_training_steps_for_scheduler = (
        args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
    )
    params_to_optimize = [
        {
            "params": list(filter(lambda p: p.requires_grad, query_model.parameters())),
            "lr": args.learning_rate,
        },
    ]
    optimizer = AdamW(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our accelerator
    query_model, doc_model, optimizer, train_dataloader, lr_scheduler = (
        accelerator.prepare(
            query_model, doc_model, optimizer, train_dataloader, lr_scheduler
        )
    )

    # The size of the training dataloader may have changed due to accelerator.prepare, so we need to recalculate our total training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    if num_training_steps_for_scheduler != max_train_steps:
        logger.warning(
            f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
            f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
            f"This inconsistency may result in the learning rate scheduler not functioning properly."
        )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Report
    if accelerator.is_main_process and args.do_report:
        tracker_name = args.tracker_name
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Log some info about our training
    total_batch_size = (
        per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"Num epochs = {num_train_epochs}")
    logger.info(f"Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(
        f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Train!
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, num_train_epochs):
        query_model.train()
        for _, batch in enumerate(train_dataloader):
            models_to_accumulate = [query_model]
            with accelerator.accumulate(models_to_accumulate):
                queries = batch["query"]
                docs = batch["text"]
                loss = genx_transformer(queries, docs)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(query_model.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

        # Validate!
        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                # log_validation(...)
                ...  # TODO: Add validation here

    # Evaluate!
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ...  # TODO: Final evaluation and push to HF if needed

    # Finish
    accelerator.end_training()


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments))
    (args,) = parser.parse_args_into_dataclasses()
    ddp_process(args)

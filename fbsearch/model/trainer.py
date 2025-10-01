import itertools
import logging
import math
import os
from dataclasses import dataclass, field
from functools import partial

import pandas as pd
import transformers.utils.logging as transformers_utils_logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    gather_object,
    ProjectConfiguration,
    set_seed,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import get_scheduler, HfArgumentParser
from transformers.trainer_utils import seed_worker

from ..store.retrieval import get_sample_datasets, PrefixTreeStore
from .llama import FBSearchTransformer, get_fbsearch_transformer

DEBUG = os.environ.get("DEBUG", False)

logger = get_logger(__name__)


@dataclass
class FBSearchTrainingArguments:
    # Model
    doc_model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    query_model_name_or_path: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    model_max_length: int = field(default=2048)

    # Prompt
    doc_prompt_before: str = field(default="Document: ")
    doc_prompt_after: str = field(
        default="Task: Generate a summary with several words. Directly say the words without explanation."
    )
    query_prompt_before: str = field(default="Query: ")
    query_prompt_after: str = field(
        default="Task: Guess you answer with several words. Directly say the words without explanation."
    )

    # Accelerator
    mixed_precision: str = field(default="bf16")
    report_to: str = field(default="wandb")
    do_report: bool = field(default=True)

    # Seed
    seed: int = field(default=0)

    # Logging
    output_dir: str = field(default="runs")
    logging_dir: str = field(default="logs")
    tracker_name: str = field(default="supertiny")
    run_tags: list[str] = field(default_factory=list)
    run_name: str = field(default="many2many")

    # Store
    num_next_tokens: int = field(default=5)
    num_beams_doc: int = field(default=1)
    num_beams_query: int = field(default=2)
    insertion_depth: int = field(default=5)
    caching: bool = field(default=True)

    # Dataloader
    dataloader_num_workers: int = field(default=4)

    # Training
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=5)
    lr_warmup_steps: int = field(default=0)
    lr_scheduler: str = field(default="cosine")
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: int = field(default=10)
    learning_rate: float = field(default=2e-5)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)
    adam_weight_decay: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    validation_epochs: int = field(default=1)

    # Resume checkpoint
    resume_from_checkpoint: str = field(default=None)
    checkpointing_epochs: int = field(default=99999)


class FBSearchTrainer:
    def __init__(
        self,
        transformer: FBSearchTransformer,
        args: FBSearchTrainingArguments,
        corpus_dataset: Dataset,
        query2doc_dataset: Dataset,
        query_train_dataset: Dataset = None,
        query_dev_dataset: Dataset = None,
        corpus_collate_fn=None,
        query2doc_collate_fn=None,
        query_collate_fn=None,
    ):
        self.transformer = transformer
        self.args = args

        self.corpus_dataset = corpus_dataset
        self.query2doc_dataset = query2doc_dataset
        self.query_train_dataset = query_train_dataset
        self.query_dev_dataset = query_dev_dataset

        self.corpus_collate_fn = corpus_collate_fn
        self.query2doc_collate_fn = query2doc_collate_fn
        self.query_collate_fn = query_collate_fn

    def init_accelerator(self):
        # Arguments
        args = self.args

        # Logging dir
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=logging_dir
        )
        kwargs = DistributedDataParallelKwargs()

        # Init accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
            device_placement=True,  # Ensure proper device placement
        )
        # Make one log on every process with the configuration for debugging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        transformers_utils_logging.set_verbosity_error()

        self.accelerator = accelerator

        # Process-aware seeding
        set_seed(args.seed + accelerator.process_index)
        return accelerator

    def _get_train_sampler(self, query2doc_dataset) -> RandomSampler:
        return RandomSampler(query2doc_dataset)

    def _get_eval_sampler(self, query_dataset) -> SequentialSampler | None:
        accelerator = self.accelerator
        if accelerator.num_processes <= 1:
            return SequentialSampler(query_dataset)
        else:
            return None

    def _get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler_fn=None,
        is_training: bool = False,
        collate_fn=None,
    ):
        args = self.args
        accelerator = self.accelerator

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "num_workers": args.dataloader_num_workers,
        }
        if sampler_fn is not None:
            dataloader_params["sampler"] = sampler_fn(dataset)
        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=args.dataloader_num_workers,
                rank=accelerator.process_index,
            )

        dataloader = accelerator.prepare(DataLoader(dataset, **dataloader_params))
        return dataloader

    def get_query2doc_dataloader(self) -> DataLoader:
        args = self.args
        if self.query2doc_dataset is None:
            raise ValueError("Trainer: training requires a query2doc_dataset.")

        return self._get_dataloader(
            dataset=self.query2doc_dataset,
            batch_size=args.per_device_train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
            collate_fn=self.query2doc_collate_fn,
        )

    def get_query_dataloader(self, query_dataset) -> DataLoader:
        args = self.args

        return self._get_dataloader(
            dataset=query_dataset,
            batch_size=args.per_device_eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            is_training=False,
            collate_fn=self.query_collate_fn,
        )

    def get_corpus_dataloader(self) -> DataLoader:
        args = self.args
        if self.corpus_dataset is None:
            raise ValueError("Trainer: evaluation requires a corpus_dataset.")

        return self._get_dataloader(
            dataset=self.corpus_dataset,
            batch_size=args.per_device_eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            is_training=False,
            collate_fn=self.corpus_collate_fn,
        )

    def train(self, query_datasets: dict[str | Dataset] = None):
        args: FBSearchTrainingArguments = self.args
        self.plotted = False

        # Make a directory for outputs
        os.makedirs(args.output_dir, exist_ok=True)

        # Init accelerator
        accelerator: Accelerator = self.init_accelerator()

        # Datasets for train (query-> doc), and eval (query-> doc)
        self.train_dataloader = self.get_query2doc_dataloader()

        # Create optimizer and scheduler
        train_configs = self.create_optimizer_and_scheduler()
        self.init_trackers()
        self.logger_info_dict(train_configs)

        max_train_steps = train_configs["max_train_steps"]
        num_train_epochs = train_configs["num_train_epochs"]
        checkpointing_steps = train_configs["checkpointing_steps"]
        num_update_steps_per_epoch = train_configs["num_update_steps_per_epoch"]

        # Resume from checkpoint
        global_step, initial_global_step, first_epoch = self.resume_from_checkpoint(
            num_update_steps_per_epoch
        )

        # Train
        progress_bar = tqdm(
            range(0, max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        epoch = first_epoch
        for epoch in range(first_epoch, num_train_epochs):
            self.transformer.query_model.train()
            for _, batch in enumerate(self.train_dataloader):
                models_to_accumulate = [self.transformer.query_model]
                with accelerator.accumulate(models_to_accumulate):
                    loss, logs = self.training_step(batch)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % checkpointing_steps == 0:
                            save_path = os.path.join(
                                args.output_dir, f"checkpoint-{global_step}"
                            )
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= max_train_steps:
                    break

            # Validate
            if epoch % args.validation_epochs == 0:
                if query_datasets is None:
                    query_datasets = {}
                    if self.query_train_dataset is not None:
                        query_datasets["train"] = self.query_train_dataset
                    if self.query_dev_dataset is not None:
                        query_datasets["dev"] = self.query_dev_dataset

                for flag, query_dataset in query_datasets.items():
                    self.evaluate(
                        query_dataset,
                        flag=flag,
                        global_step=global_step,
                    )

        # End training
        accelerator.wait_for_everyone()
        accelerator.end_training()

    def evaluate(self, query_dataset, flag: str, **kwargs):
        args = self.args
        global_step = kwargs.get("global_step", None)
        accelerator: Accelerator = self.accelerator

        if not hasattr(self, "corpus_dataloader"):
            self.corpus_dataloader = self.get_corpus_dataloader()

        query_dataloader = self.get_query_dataloader(query_dataset)

        if accelerator.is_main_process:
            store = PrefixTreeStore(
                self.transformer,
                insertion_depth=args.insertion_depth,
            )

        for batch in self.corpus_dataloader:
            doc_ids = batch["doc_id"]
            contents = batch["content"]
            genxs, tokens = self.transformer.index_doc(contents)

            # Gather results from all processes
            ga_doc_ids = gather_object(doc_ids)
            ga_contents = gather_object(contents)
            # need to use `gather` here in case batch size is 1
            ga_genxs = accelerator.gather(genxs)
            ga_tokens = gather_object(tokens)

            if accelerator.is_main_process:
                store.insert(
                    {"doc_id": ga_doc_ids, "content": ga_contents},
                    ga_genxs,
                    ga_tokens,
                )

        if accelerator.is_main_process:
            plot_path = os.path.join(args.output_dir, "frequencies.pdf")
            if not self.plotted:
                token_stats, stats_fig = store.plot_token_frequencies(
                    save_path=plot_path,
                )
                self.plotted = True

                # Find wandb tracker and log table
                wandb_tracker = next(
                    (t for t in self.accelerator.trackers if t.name == "wandb"), None
                )
                if wandb_tracker:
                    import wandb

                    wandb_tracker.log(
                        {
                            "token_freq_fig": wandb.Image(
                                stats_fig, caption="Token Frequency"
                            )
                        },
                        commit=False,  # Don't create/advance to new step
                    )

        if accelerator.is_main_process:
            all_doc_ids = []
            all_results = []

        for batch in query_dataloader:
            doc_ids = batch["doc_id"]
            queries = batch["query"]
            genxs, tokens = self.transformer.index_query(queries)

            # Gather results from all processes
            ga_doc_ids = gather_object(doc_ids)
            ga_queries = gather_object(queries)
            ga_genxs = accelerator.gather(genxs)
            ga_tokens = gather_object(tokens)

            if accelerator.is_main_process:
                ga_results = store.query(ga_queries, ga_genxs, ga_tokens)
                all_doc_ids += ga_doc_ids
                all_results += ga_results

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            all_results, metrics = store.results2metrics(all_doc_ids, all_results)
            log_metrics = {f"{flag}:{k}": v for k, v in metrics.items()}
            self.accelerator.log(log_metrics, step=global_step)

            metrics = pd.DataFrame([metrics])
            metrics.to_csv(
                os.path.join(args.output_dir, f"{flag}-metrics-{global_step}.csv"),
                index=False,
            )

    def resume_from_checkpoint(self, num_update_steps_per_epoch):
        args = self.args
        accelerator = self.accelerator

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
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])

                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            global_step = 0
            initial_global_step = 0
            first_epoch = 0

        return global_step, initial_global_step, first_epoch

    def create_optimizer_and_scheduler(self):
        args = self.args
        accelerator = self.accelerator

        # Move models to GPU
        self.transformer.to(accelerator.device)
        self.transformer.doc_model.requires_grad_(False)

        # Optimizer and learning rate scheduler
        num_warmup_steps_for_scheduler = (
            args.lr_warmup_steps * accelerator.num_processes
        )

        # Here, train_dataloader has already been prepared by the accelerator,
        # otherwise, should do math.ceil(len(self.train_dataloader) / accelerator.num_processes)
        len_train_dataloader_after_sharding = len(self.train_dataloader)
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * accelerator.num_processes
            * num_update_steps_per_epoch
        )
        params_to_optimize = [
            {
                "params": list(
                    filter(
                        lambda p: p.requires_grad,
                        self.transformer.query_model.parameters(),
                    )
                ),
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
            scheduler_specific_kwargs={"num_cycles": 0.5},
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Prepare models, optimizer, and scheduler for training
        (
            self.transformer.query_model,
            self.transformer.doc_model,
            self.optimizer,
            self.lr_scheduler,
        ) = accelerator.prepare(
            self.transformer.query_model,
            self.transformer.doc_model,
            self.optimizer,
            self.lr_scheduler,
        )

        # The size of the training dataloader may have changed due to accelerator.prepare, so we need to recalculate our total training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / args.gradient_accumulation_steps
        )
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(self.train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        checkpointing_steps = args.checkpointing_epochs * num_update_steps_per_epoch

        train_configs = {
            "num_examples": len(self.train_dataloader.dataset),
            "num_batches_per_epoch": len(self.train_dataloader),
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "total_batch_size": total_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_update_steps_per_epoch": num_update_steps_per_epoch,
            "max_train_steps": max_train_steps,
            "checkpointing_epochs": args.checkpointing_epochs,
            "validation_epochs": args.validation_epochs,
            "checkpointing_steps": checkpointing_steps,
        }
        return train_configs

    def init_trackers(self):
        args = self.args
        accelerator = self.accelerator
        if accelerator.is_main_process and args.do_report:
            tracker_name = args.tracker_name
            accelerator.init_trackers(
                tracker_name,
                config=vars(self.args),
                init_kwargs={"wandb": {"name": args.run_name, "tags": args.run_tags}},
            )

    def logger_info_dict(self, dict):
        for key, value in dict.items():
            logger.info(f"{key}: {value}")

    def compute_loss(self, batch):
        query = batch["query"]
        content = batch["content"]
        loss = self.transformer.compute_loss(query, content)
        return loss

    def training_step(self, batch):
        args = self.args
        accelerator = self.accelerator

        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        accelerator.backward(loss)

        grad_norm = None
        if accelerator.sync_gradients:
            params_to_clip = itertools.chain(self.transformer.query_model.parameters())
            grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()

        # Create logs with gradient norm
        logs = {
            "loss": loss.detach().item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

        # Add gradient norm if available
        if grad_norm is not None:
            logs["grad_norm"] = grad_norm.item()

        return loss, logs


if __name__ == "__main__":
    # `torchrun --nproc_per_node=2 -m fbsearch.model.trainer --tracker_name supertiny --run_name many2many --run_tags many2many tiny`
    # `torchrun --nproc_per_node=2 -m fbsearch.model.trainer --tracker_name supertiny --run_name one2one --run_tags one2one tiny`

    parser = HfArgumentParser((FBSearchTrainingArguments))
    (args,) = parser.parse_args_into_dataclasses()

    if args.run_name.startswith("many2many"):
        query_type = "many2many"
    elif args.run_name.startswith("one2one"):
        query_type = "one2one"
    datas = get_sample_datasets(query_type=query_type)
    query2doc_dataset = datas["query2doc_dataset"]
    query2doc_collate_fn = datas["query2doc_collate_fn"]
    query_dataset = datas["query_dataset"]
    query_collate_fn = datas["query_collate_fn"]
    corpus_dataset = datas["corpus_dataset"]
    corpus_collate_fn = datas["corpus_collate_fn"]

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
        query_train_dataset=query_dataset,
        query_dev_dataset=None,
        # Collate functions
        corpus_collate_fn=corpus_collate_fn,
        query2doc_collate_fn=query2doc_collate_fn,
        query_collate_fn=query_collate_fn,
    )
    trainer.train()

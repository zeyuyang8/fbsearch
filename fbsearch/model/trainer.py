import logging
import os
from dataclasses import dataclass, field
from functools import partial

import transformers.utils.logging as transformers_utils_logging

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import HfArgumentParser
from transformers.trainer_utils import seed_worker

from .llama import FBSearchTransformer

DEBUG = os.environ.get("DEBUG", False)

logger = get_logger(__name__)


@dataclass
class FBSearchArguments:
    # Accelerator
    mixed_precision: str = field(default="bf16")
    report_to: str = field(default="wandb")
    do_report: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)
    seed: int = field(default=0)
    output_dir: str = field(default="output")
    logging_dir: str = field(default="logs")

    # Dataloader
    dataloader_num_workers: int = field(default=4)
    train_batch_size: int = field(default=1)
    eval_batch_size: int = field(default=1)


class FBSearchTrainer:
    def __init__(
        self,
        transformer: FBSearchTransformer = None,
        args: FBSearchArguments = None,
        train_dataset=None,
        eval_dataset=None,
        collate_fn=None,
        store=None,
    ):
        self.transformer = transformer
        self.args = args

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn

        self.store = store

    def init_accelerator(self):
        # Logging dir
        logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        # Init accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
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
        if accelerator.is_local_main_process:
            transformers_utils_logging.set_verbosity_warning()
        else:
            transformers_utils_logging.set_verbosity_error()

        self.accelerator = accelerator

        # Process-aware seeding
        set_seed(self.args.seed + self.accelerator.process_index)

        return accelerator

    def _get_train_sampler(self, train_dataset) -> RandomSampler:
        return RandomSampler(train_dataset)

    def _get_eval_sampler(self, eval_dataset) -> SequentialSampler | None:
        if self.accelerator.num_processes <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None

    def _get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler_fn=None,
        is_training: bool = False,
    ):
        collate_fn = self.collate_fn
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": collate_fn,
            "num_workers": self.args.dataloader_num_workers,
        }
        if sampler_fn is not None:
            dataloader_params["sampler"] = sampler_fn(dataset)
        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.accelerator.process_index,
            )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
        return dataloader

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def get_eval_dataloader(self) -> DataLoader:
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a eval_dataset.")

        return self._get_dataloader(
            dataset=self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            is_training=False,
        )

    def train(self):
        # Init accelerator
        self.init_accelerator()

        # Datasets for corpus (doc), train (query-> doc), and eval (query-> doc)
        train_dataloader = self.get_train_dataloader()
        if self.eval_dataset is not None:
            eval_dataloader = self.get_eval_dataloader()

        # End training
        self.accelerator.end_training()

    def log(self):
        pass

    def create_optimizer_and_scheduler(self):
        pass

    def compute_loss(self):
        pass

    def training_step(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def prediction_step(self):
        pass


if __name__ == "__main__":
    print("Running `torchrun --nproc_per_node=2 -m fbsearch.model.trainer`")
    from ..store.retrieval import get_sample_query2doc_dataset

    train_dataset = get_sample_query2doc_dataset()
    parser = HfArgumentParser((FBSearchArguments))
    (args,) = parser.parse_args_into_dataclasses()
    trainer = FBSearchTrainer(
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore", module="torch.distributed.distributed_c10d")

TRANSFORMERS_PATH_MAP = {
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

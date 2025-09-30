import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore", module="torch.distributed.distributed_c10d")

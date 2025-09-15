# Generative Index

## Setup

If on a devserver:

```bash
with-proxy conda create --name fbs python=3.12
conda activate fbs
with-proxy conda install pip
with-proxy pip install -r requirements.txt
with-proxy pip install -e . && rm -rf ./genx.egg-info
```

Also set environment variable (CUDA version should be 12.8 if using the latest version of PyTorch):

```bash
export CUDA_HOME=...
export LD_LIBRARY_PATH=...
export PATH=$CUDA_HOME/bin:$PATH

export WANDB_API_KEY=...
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=...
export http_proxy=...
export https_proxy=...
```

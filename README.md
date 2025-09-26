# Generative Index

## Setup

If on a devserver:

```bash
with-proxy conda create --name fbsearch python=3.12
conda activate fbsearch
with-proxy conda install pip
with-proxy pip install -r requirements.txt
with-proxy pip install -e . && rm -rf ./fbsearch.egg-info
```

Or alternatively you can use:

```bash
conda activate /home/zy45/.conda/envs/fbsearch
```

Also set environment variable (CUDA version should be 12.8 if using the latest version of PyTorch):

```
nano ~/.bashrc
```

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="/usr/local/fbcode/platform010/lib/cuda-no-rpath-12.8:$LD_LIBRARY_PATH"
export PATH=$CUDA_HOME/bin:$PATH

export WANDB_API_KEY="..."
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="..."
export http_proxy=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080
export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fb"
```


Convert your notebook to script:

```bash
jupyter nbconvert --to python debug.ipynb --PythonExporter.exclude_markdown=True --TagRemovePreprocessor.remove_cell_tags="['notebook_only']" --log-level ERROR
```

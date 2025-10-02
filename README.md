# Generative Index

## Setup

If on a devserver:

```bash
with-proxy conda create --name fbsearch python=3.12
conda activate fbsearch
with-proxy conda install pip
with-proxy pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
with-proxy pip install -e . && rm -rf ./fbsearch.egg-info
```

Test if torch CUDA is installed:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Or alternatively you can use:

```bash
conda activate /home/zy45/.conda/envs/fbsearch
```

Also set environment variable (CUDA version should be 12.8 if using the latest version of PyTorch):

```bash
nano ~/.bashrc
```

Then add the following lines and save:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="/usr/local/fbcode/platform010/lib/cuda-no-rpath-12.8:$LD_LIBRARY_PATH"
export PATH=$CUDA_HOME/bin:$PATH

export WANDB_API_KEY="..."
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="..."
```

Before running a shell script, make sure to set the environment variable at the beginning of the script:

```bash
#!/bin/bash

export http_proxy=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080
export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fb"
```

Do not set the above environment variables in your `.bashrc` file, as it will cause some network requests to fail.

## Useful Commands

### Jupyter

Convert your notebook to script:

```bash
jupyter nbconvert --to python notebook.ipynb --PythonExporter.exclude_markdown=True --TagRemovePreprocessor.remove_cell_tags="['notebook_only']" --log-level ERROR
```

### Avoiding `BUCK`

Run this so that you don't have to run `buck build` every time you make a change to the code:

```bash
find . -name "*.py" -exec grep -L "@noautodeps" {} \; | while read file; do
    sed -i '1i# @noautodeps' "$file"
done
```

### Setup Devserver/MAST/Manifold

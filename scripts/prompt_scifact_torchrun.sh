#!/usr/bin/bash

torchrun --nnodes=1 --nproc_per_node=4 ddp_prompt.py

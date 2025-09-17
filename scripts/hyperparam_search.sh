#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python hyperparam_search.py --nproc_per_node=2

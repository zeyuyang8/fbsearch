#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

torchrun --nproc_per_node=2 --master-port 9999 train_scifact.py \
    --tracker_name anshu \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --num_beams 1 \
    --num_next_tokens 1 \
    --insertion_depth 1

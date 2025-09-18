#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

torchrun --nproc_per_node=2 --master-port 1111 train_scifact.py \
    --tracker_name debug \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --num_beams 1 \
    --num_next_tokens 1 \
    --insertion_depth 1

#!/bin/bash

torchrun --nproc_per_node=4 train_scifact.py \
    --tracker_name scifact \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --num_beams 10 \
    --num_next_tokens 5 \
    --insertion_depth 3

num_train_epochs_list=(10 20)
num_beams_list=(1 3 5 10)

for num_train_epochs in "${num_train_epochs_list[@]}"; do
    for num_beams in "${num_beams_list[@]}"; do
        torchrun --nproc_per_node=4 train_scifact.py \
            --tracker_name scifact \
            --learning_rate 1e-4 \
            --num_train_epochs "$num_train_epochs" \
            --num_beams "$num_beams" \
            --num_next_tokens 5 \
            --insertion_depth 4
    done
done

#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export http_proxy=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080
export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fb"

torchrun --nproc_per_node=2 train_scifact.py \
    --tracker_name mteb \
    --run_name scifact \
    --output_dir ../runs/scifact \
    --num_beams_query 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --lr_warmup_steps 50 \
    --num_train_epochs 20

#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2

torchrun --nproc_per_node=2 --master-port 2025 train.py \
    --do_report true \
    --corpus_filename corpus.jsonl \
    --train_queries_filename claims_train.jsonl \
    --dev_queries_filename claims_dev.jsonl \
    --store_state_filename state.joblib \
    --load_store_state true \
    --output_dir runs/scifact/train-on-real-3 \
    --tracker_name sep19 \
    --learning_rate 5e-5 \
    --validation_epochs 5 \
    --num_train_epochs 200 \
    --num_beams 1 \
    --num_next_tokens 3 \
    --insertion_depth 3

torchrun --nproc_per_node=2 --master-port 2026 train.py \
    --do_report true \
    --corpus_filename corpus.jsonl \
    --train_queries_filename claims_train.jsonl \
    --dev_queries_filename claims_dev.jsonl \
    --store_state_filename state.joblib \
    --load_store_state true \
    --output_dir runs/scifact/train-on-real-5 \
    --tracker_name sep19 \
    --learning_rate 5e-5 \
    --validation_epochs 5 \
    --num_train_epochs 200 \
    --num_beams 1 \
    --num_next_tokens 5 \
    --insertion_depth 5

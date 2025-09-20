#!/bin/bash

NUM_BEAMS=(1)
NUM_NEXT_TOKENS=(3 5)

for NUM_BEAM in "${NUM_BEAMS[@]}"
do
    for NUM_NEXT_TOKEN in "${NUM_NEXT_TOKENS[@]}"
    do
        echo "NUM_BEAM: $NUM_BEAM"
        echo "NUM_NEXT_TOKEN: $NUM_NEXT_TOKEN"
        OUTPUT_DIR=runs/scifact/train-real-beam"$NUM_BEAM"-next"$NUM_NEXT_TOKEN"
        echo "OUTPUT_DIR: $OUTPUT_DIR"

        torchrun --nproc_per_node=4 --master-port 2025 train.py \
            --do_report true \
            --corpus_filename corpus.jsonl \
            --train_queries_filename claims_train.jsonl \
            --dev_queries_filename claims_dev.jsonl \
            --load_store_state true \
            --tracker_name sep19 \
            --learning_rate 5e-5 \
            --num_train_epochs 20 \
            --validation_epochs 1 \
            --num_beams "$NUM_BEAM" \
            --num_next_tokens "$NUM_NEXT_TOKEN" \
            --insertion_depth "$NUM_NEXT_TOKEN" \
            --output_dir "$OUTPUT_DIR"
    done
done

#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export http_proxy=http://fwdproxy:8080
export https_proxy=http://fwdproxy:8080
export no_proxy=".fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fb"

RUN_NAME_OPTIONS=("many2many" "one2one")

for RUN_NAME in "${RUN_NAME_OPTIONS[@]}"; do

torchrun --nproc_per_node=2 -m fbsearch.model.trainer \
    --tracker_name supertiny \
    --run_name "$RUN_NAME" --run_tags "$RUN_NAME" tiny \
    --do_report true \
    --output_dir ./runs/tiny-"$RUN_NAME"

done

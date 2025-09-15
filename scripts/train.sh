#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --config_file ./cfg.yaml \
  --num_processes 2 \
  train_scifact_accelerate.py

#!/bin/bash

torchrun --nproc_per_node=2 train_scifact_accelerate.py

#!/bin/bash

torchrun --nproc_per_node=4 train_scifact.py

#!/usr/bin/bash

# Here are 2 variables to configure
export TORCHXCONFIG="./.torchxconfig.h100"  # ? a100 and h100 can be used interchangeably here
export NAME="h100test"  # ? Configure this to be the name of the job

export HTYPE="grandteton"
# Based on the configuration, determine the hardware type
if [[ "$TORCHXCONFIG" == *"a100"* ]]; then
    export HTYPE="zionex_80g"
elif [[ "$TORCHXCONFIG" == *"h100"* ]]; then
    export HTYPE="grandteton"
fi

# Run the MAST job - component args BEFORE --, script args AFTER --
with-proxy torchx run mast.py:train \
    --h $HTYPE \
    --nnodes 2 \
    --nproc_per_node 8 \
    --script train.py \
    --name $NAME \
    -- \
    --gradient_accumulation_steps=1  # ? Configure this for your train scripts

#!/usr/bin/bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Edited run.sh script to run mount.sh
if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
    source /packages/torchx_conda_mount/mount.sh
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" &&
python3 "$@"

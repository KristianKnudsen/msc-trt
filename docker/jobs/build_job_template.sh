#!/bin/bash

BASE="/workspace/trt-scripts"
MODEL_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
CONFIG="build_config_W16A16_LOGITS.json"

python $BASE/model/build_engine.py \
  --config $BASE/model/configs/$CONFIG \
  --base $BASE \
  --model-dir $MODEL_DIR

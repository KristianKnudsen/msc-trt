#!/bin/bash

BASE="/workspace/trt-scripts"
MODEL_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
MODEL="Qwen25_3B"
QUANT="W16A16"
TASK="mmlu"
CONFIG="eval_${TASK}.json"

python $BASE/eval/custom_lmeval_wrapper.py \
  --config $BASE/eval/configs/tasks/$CONFIG \
  --base $BASE \
  --engine-dir $MODEL/${QUANT}_LOGITS \
  --model-dir $MODEL_DIR

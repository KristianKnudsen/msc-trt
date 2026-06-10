#!/bin/bash

BASE="/workspace/trt-scripts"
MODEL_DIR="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
QUANT="W16A16"

CONFIGS=(
  "eval_humaneval.json"
  "eval_mbpp.json"
  "eval_gpqa.json"
  "eval_mmlu.json"
  "eval_hellaswag.json"
  "eval_winogrande.json"
  "eval_gsm8k.json"
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running $CONFIG..."
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/tasks/$CONFIG \
    --base $BASE \
    --engine-dir Qwen25_3B/${QUANT}_LOGITS \
    --model-dir $MODEL_DIR
done

#!/bin/bash
set -euo pipefail

BASE="/workspace/trt-scripts"
MODEL="Qwen25_14B"
TOKENIZER="Qwen25"
CONFIG="base_config.json"

QUANTS=(
  "W8A16"
  "W8A16_INT8KV"
  "W8A8_SQ"
  "W8A8_SQ_INT8KV"
  "W4A16"
  "W4A16_INT8KV"
  "W4A16_AWQ"
  "W4A16_AWQ_FP8KV"
)

DATASETS=(
  "Qwen_Synthetic_short_in_short_out_256_256_16_16.txt"
  "Qwen_Synthetic_long_in_short_out_2560_256_128_16.txt"
  "Qwen_Synthetic_short_in_long_out_256_2560_16_128.txt"
  "Qwen_Synthetic_long_in_long_out_2560_2560_128_128.txt"
)

for QUANT in "${QUANTS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    python "$BASE/bench/bench_latency.py" \
      --base "$BASE" \
      --engine "$MODEL/$QUANT" \
      --dataset "$DATASET" \
      --config "$CONFIG" \
      --tokenizer "$TOKENIZER"
  done
done

#!/bin/bash
set -euo pipefail

TOKENIZER="/workspace/trt-scripts/model/tokenizers/Mistral"
PREPARE_DATASET="/workspace/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py"
OUTDIR="/workspace/trt-scripts/bench/datasets"
NUM_REQUESTS=25

# name input_mean output_mean input_stdev output_stdev
WORKLOADS=(
  "long_in_long_out 2560 2560 128 128"
  "short_in_short_out 256 256 16 16"
  "long_in_short_out 2560 256 128 16"
  "short_in_long_out 256 2560 16 128"
)

for WORKLOAD in "${WORKLOADS[@]}"; do
  read -r NAME INPUT_MEAN OUTPUT_MEAN INPUT_STDEV OUTPUT_STDEV <<< "$WORKLOAD"

  python "$PREPARE_DATASET" \
    --stdout \
    --tokenizer "$TOKENIZER" \
    token-norm-dist \
    --input-mean "$INPUT_MEAN" \
    --output-mean "$OUTPUT_MEAN" \
    --input-stdev "$INPUT_STDEV" \
    --output-stdev "$OUTPUT_STDEV" \
    --num-requests "$NUM_REQUESTS" \
    > "$OUTDIR/Mistral_Synthetic_${NAME}_${INPUT_MEAN}_${OUTPUT_MEAN}_${INPUT_STDEV}_${OUTPUT_STDEV}.txt"
done

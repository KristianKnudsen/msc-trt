#!/bin/bash
# Submit eval jobs for all Mistral quants

DIR="$(dirname "$(realpath "$0")")"
SCRIPT="$DIR/eval_array.sh"

QUANTS=(
  "W16A16"
  "W8A16"
  "W8A8_SQ"
  "W4A16"
  "W4A16_AWQ"
  "W16A16_INT8KV"
  "W8A16_INT8KV"
  "W8A8_SQ_INT8KV"
  "W4A16_INT8KV"
  "W4A16_AWQ_FP8KV"
)

for QUANT in "${QUANTS[@]}"; do
  JOB_NAME="eval_mistral_${QUANT}"

  # humaneval(0), mbpp(1), winogrande(4), gpqa(5), gsm8k(6)
  sbatch --export=ALL,QUANT="$QUANT" --job-name="$JOB_NAME" \
    --array=0,1,4,5,6 --time=00:31:15 "$SCRIPT"

  # hellaswag(3)
  sbatch --export=ALL,QUANT="$QUANT" --job-name="$JOB_NAME" \
    --array=3 --time=02:30:00 "$SCRIPT"

  # mmlu(2)
  sbatch --export=ALL,QUANT="$QUANT" --job-name="$JOB_NAME" \
    --array=2 --time=05:00:00 --mem=64G "$SCRIPT"


  # wikitext(7)
  sbatch --export=ALL,QUANT="$QUANT" --job-name="$JOB_NAME" \
    --array=7 --time=00:32:30 "$SCRIPT"
done

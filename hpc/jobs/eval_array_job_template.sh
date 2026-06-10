#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_<QUANT>"
#SBATCH -c2
#SBATCH --time=00:12:30
#SBATCH --begin=now
#SBATCH --output=<BASE_PATH>/trt-scripts/hpc/logs/eval_%A_%a.out
# NOTE: --array and --time are set per task-group in the submit script, not here.

BASE="<BASE_PATH>/trt-scripts"
MODEL="<MODEL>"
QUANT="<QUANT>"
MODEL_DIR="~/hf-cache/models/<MODEL_DIR>"

# Indices must match --array in the submit script (0-indexed).
CONFIGS=(
  "eval_humaneval.json"   # 0
  "eval_mbpp.json"        # 1
  "eval_mmlu.json"        # 2
  "eval_hellaswag.json"   # 3
  "eval_winogrande.json"  # 4
  "eval_gpqa.json"        # 5
  "eval_gsm8k.json"       # 6
  "eval_wikitext.json"    # 7
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/tasks/$CONFIG \
    --base $BASE \
    --engine-dir "$MODEL/${QUANT}_LOGITS" \
    --model-dir $MODEL_DIR

#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_qwen_kv_cache"
#SBATCH -c2
#SBATCH --time=00:20:00
#SBATCH --begin=now
#SBATCH --array=0-14
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/eval_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL="Qwen25_3B"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"

QUANTS=(
  "W4A16_AWQ_FP8KV"
  "W4A16_INT8KV"
  "W8A16_INT8KV"
  "W16A16_FP8KV"
  "W16A16_INT8KV"
)

CONFIGS=(
  "eval_humaneval.json"
  "eval_mbpp.json"
  "eval_gsm8k.json"
)

TASK_COUNT=${#CONFIGS[@]}
QUANT_INDEX=$((SLURM_ARRAY_TASK_ID / TASK_COUNT))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % TASK_COUNT))
QUANT=${QUANTS[$QUANT_INDEX]}
CONFIG=${CONFIGS[$CONFIG_INDEX]}

if [[ -z "${QUANT:-}" || -z "${CONFIG:-}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/tasks/$CONFIG \
    --base $BASE \
    --engine-dir ${MODEL}/${QUANT}_LOGITS \
    --model-dir $MODEL_DIR

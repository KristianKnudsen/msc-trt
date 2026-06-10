#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="build_<MODEL>"
#SBATCH -c4
#SBATCH --time=00:30:00
#SBATCH --begin=now
#SBATCH --array=0-N
#SBATCH --output=<BASE_PATH>/trt-scripts/hpc/logs/build_%A_%a.out
# NOTE: Update --array=0-N to match the number of configs (0-indexed).

BASE="<BASE_PATH>/trt-scripts"
MODEL="<MODEL>"
MODEL_DIR="~/hf-cache/models/<MODEL_DIR>"

CONFIGS=(
  "build_config_<QUANT>_LOGITS.json"  # 0
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/$MODEL/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR

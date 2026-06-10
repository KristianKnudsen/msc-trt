#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="build_mistral_kv_cache"
#SBATCH -c4
#SBATCH --time=00-00:30:00
#SBATCH --begin=now
#SBATCH --array=0-6
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/build_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL="Mistral_7B_v0_3"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Mistral-7B-v0.3"

CONFIGS=(
  "W4A16_AWQ_FP8KV/build_config_W4A16_AWQ_FP8KV_LOGITS.json"            # 0
  "W4A16_FP8KV/build_config_W4A16_FP8KV_LOGITS.json"                    # 1
  "W4A16_INT8KV/build_config_W4A16_INT8KV_LOGITS.json"                  # 2
  "W8A16_FP8KV/build_config_W8A16_FP8KV_LOGITS.json"                    # 3
  "W8A16_INT8KV/build_config_W8A16_INT8KV_LOGITS.json"                  # 4
  "W16A16_FP8KV/build_config_W16A16_FP8KV_LOGITS.json"                  # 5
  "W16A16_INT8KV/build_config_W16A16_INT8KV_LOGITS.json"                # 6
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/$MODEL/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR

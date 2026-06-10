#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --job-name="build_qwen_7B_kv_cache"
#SBATCH -c2
#SBATCH --time=00-00:30:00
#SBATCH --begin=now
#SBATCH --array=0-6
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/build_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-7B"

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
    --config $BASE/model/configs/hpc/Qwen25_7B/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR

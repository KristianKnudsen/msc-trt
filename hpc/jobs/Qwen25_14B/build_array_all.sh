#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name="build_qwen_14B"
#SBATCH -c2
#SBATCH --time=00-00:30:00
#SBATCH --begin=now
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/build_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-14B"

CONFIGS=(
  "W16A16/build_config_W16A16_LOGITS.json"                     # 0
  "W8A16/build_config_W8A16_LOGITS.json"                       # 1
  "W8A8_SQ/build_config_W8A8_SQ_LOGITS.json"                   # 2
  "W4A16/build_config_W4A16_LOGITS.json"                       # 3
  "W4A16_AWQ/build_config_W4A16_AWQ_LOGITS.json"               # 4
  "W16A16_INT8KV/build_config_W16A16_INT8KV_LOGITS.json"       # 5
  "W8A16_INT8KV/build_config_W8A16_INT8KV_LOGITS.json"         # 6
  "W8A8_SQ_INT8KV/build_config_W8A8_SQ_INT8KV_LOGITS.json"     # 7
  "W4A16_INT8KV/build_config_W4A16_INT8KV_LOGITS.json"         # 8
  "W4A16_AWQ_FP8KV/build_config_W4A16_AWQ_FP8KV_LOGITS.json"   # 9
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/Qwen25_14B/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR

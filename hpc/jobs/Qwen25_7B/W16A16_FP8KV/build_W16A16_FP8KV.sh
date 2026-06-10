#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --job-name="build_W16A16_FP8KV_7B"
#SBATCH -c2
#SBATCH --time=00-00:30:00
#SBATCH --begin=now
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/build_%j.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-7B"
CONFIG="W16A16_FP8KV/build_config_W16A16_FP8KV_LOGITS.json"

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/Qwen25_7B/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR

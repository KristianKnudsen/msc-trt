#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="build_<MODEL>_<QUANT>"
#SBATCH -c4
#SBATCH --time=00:30:00
#SBATCH --begin=now
#SBATCH --output=<BASE_PATH>/trt-scripts/hpc/logs/build_%j.out

BASE="<BASE_PATH>/trt-scripts"
MODEL="<MODEL>"
QUANT="<QUANT>"
MODEL_DIR="~/hf-cache/models/<MODEL_DIR>"
CONFIG="build_config_${QUANT}_LOGITS.json"

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/model/build_engine.py \
    --config $BASE/model/configs/hpc/$MODEL/$CONFIG \
    --base $BASE \
    --model-dir $MODEL_DIR

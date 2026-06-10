#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_<QUANT>_<TASK>"
#SBATCH -c2
#SBATCH --time=00:12:30
#SBATCH --begin=now
#SBATCH --output=<BASE_PATH>/trt-scripts/hpc/logs/eval_%j.out

BASE="<BASE_PATH>/trt-scripts"
MODEL="<MODEL>"
QUANT="<QUANT>"
TASK="<TASK>"
MODEL_DIR="~/hf-cache/models/<MODEL_DIR>"
CONFIG="eval_${TASK}.json"

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/tasks/$CONFIG \
    --base $BASE \
    --engine-dir "$MODEL/${QUANT}_LOGITS" \
    --model-dir $MODEL_DIR

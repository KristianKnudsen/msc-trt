#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_W8A8_SQ"
#SBATCH -c2
#SBATCH --time=00:12:30
#SBATCH --begin=now
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/eval_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Qwen2.5-3B"
QUANT="W8A8_SQ"

CONFIGS=(
  "eval_humaneval.json"
  "eval_mbpp.json"
  "eval_mmlu.json"
  "eval_hellaswag.json"
  "eval_winogrande.json"
  "eval_gpqa.json"
  "eval_gsm8k.json"
  "eval_wikitext.json"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun /usr/bin/apptainer exec --nv --writable-tmpfs \
  $BASE/hpc/trtllm-tools.sif \
  python $BASE/eval/custom_lmeval_wrapper.py \
    --config $BASE/eval/configs/tasks/$CONFIG \
    --base $BASE \
    --engine-dir Qwen25_3B/${QUANT}_LOGITS \
    --model-dir $MODEL_DIR

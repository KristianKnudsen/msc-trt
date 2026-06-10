#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name="eval_mistral"
#SBATCH -c2
#SBATCH --time=00:31:15
#SBATCH --begin=now
#SBATCH --output=/cluster/home/krisskn/master-thesis/trt-scripts/hpc/logs/eval_mistral_%A_%a.out

BASE="/cluster/home/krisskn/master-thesis/trt-scripts"
MODEL="Mistral_7B_v0_3"
MODEL_DIR="/cluster/home/krisskn/master-thesis/hf-cache/models/Mistral-7B-v0.3"

if [[ -z "${QUANT:-}" ]]; then
  echo "QUANT must be set, e.g. QUANT=W16A16 sbatch --array=5 eval_array.sh"
  exit 1
fi

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

if [[ -z "${CONFIG:-}" ]]; then
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

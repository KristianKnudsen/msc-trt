#!/bin/bash
SCRIPT="$(dirname "$(realpath "$0")")/eval_array_kv_cache.sh"

# Runs 7 KV-cache variants across humaneval, mbpp, and gsm8k.
sbatch "$SCRIPT"

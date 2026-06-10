#!/bin/bash
SCRIPT="$(dirname "$(realpath "$0")")/eval_array_W8A8_SQ_INT8KV.sh"

# humaneval(0), mbpp(1), winogrande(4), gpqa(5), gsm8k(6) — 12.5 min
sbatch --array=0,1,4,5,6 --time=00:20:00 "$SCRIPT"

# hellaswag(3) — 1 hour
sbatch --array=3 --time=01:36:00 "$SCRIPT"

# mmlu(2) — 2 hours, 64GB RAM
sbatch --array=2 --time=03:12:00 --mem=64G "$SCRIPT"


# wikitext(7) — 13 min
sbatch --array=7 --time=00:20:48 "$SCRIPT"

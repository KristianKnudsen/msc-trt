#!/bin/bash
# Submit eval jobs for all quants
DIR="$(dirname "$(realpath "$0")")"

bash "$DIR/W16A16/submit_eval_W16A16.sh"
bash "$DIR/W8A16/submit_eval_W8A16.sh"
bash "$DIR/W8A8_SQ/submit_eval_W8A8_SQ.sh"
bash "$DIR/W4A16/submit_eval_W4A16.sh"
bash "$DIR/W4A16_AWQ/submit_eval_W4A16_AWQ.sh"
bash "$DIR/W16A16_INT8KV/submit_eval_W16A16_INT8KV.sh"
bash "$DIR/W8A16_INT8KV/submit_eval_W8A16_INT8KV.sh"
bash "$DIR/W8A8_SQ_INT8KV/submit_eval_W8A8_SQ_INT8KV.sh"
bash "$DIR/W4A16_INT8KV/submit_eval_W4A16_INT8KV.sh"
bash "$DIR/W16A16_FP8KV/submit_eval_W16A16_FP8KV.sh"
bash "$DIR/W8A16_FP8KV/submit_eval_W8A16_FP8KV.sh"
bash "$DIR/W4A16_FP8KV/submit_eval_W4A16_FP8KV.sh"
bash "$DIR/W4A16_AWQ_FP8KV/submit_eval_W4A16_AWQ_FP8KV.sh"

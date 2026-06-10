SCRIPT="$(dirname "$(realpath "$0")")/eval_array_W4A16_AWQ_FP8KV.sh"

sbatch --array=0-7 "$SCRIPT"

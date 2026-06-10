#!/bin/bash
set -euo pipefail

BASE="${BASE:-/workspace/trt-scripts}"
MODEL="Qwen25_7B"
CONFIG_DIR="$BASE/model/configs/docker/$MODEL"
CHECKPOINT_ROOT="$BASE/model/trt_checkpoints/$MODEL"

VARIANTS=(
  "W16A16"
  "W16A16_INT8KV"
  "W8A16"
  "W8A16_INT8KV"
  "W8A8_SQ"
  "W8A8_SQ_INT8KV"
  "W4A16"
  "W4A16_INT8KV"
  "W4A16_AWQ"
  "W16A16_FP8KV"
  "W4A16_AWQ_FP8KV"
)

echo "Building all $MODEL Docker engines from checkpoints"
echo "Base:        $BASE"
echo "Configs:     $CONFIG_DIR"
echo "Checkpoints: $CHECKPOINT_ROOT"
echo

for VARIANT in "${VARIANTS[@]}"; do
  CONFIG="$CONFIG_DIR/build_existing_$VARIANT"
  CHECKPOINT_DIR="$CHECKPOINT_ROOT/$VARIANT"

  if [[ ! -f "$CONFIG" ]]; then
    echo "Missing config: $CONFIG" >&2
    exit 1
  fi

  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Missing checkpoint directory: $CHECKPOINT_DIR" >&2
    exit 1
  fi

  echo "==> Building $MODEL/$VARIANT"
  python "$BASE/model/build_engine.py" \
    --config "$CONFIG" \
    --base "$BASE"
  echo
done

echo "Done building all $MODEL engines."

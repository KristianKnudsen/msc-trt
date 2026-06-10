# model/

Contains everything for converting, quantizing, and compiling TensorRT-LLM engines from HuggingFace models.

### Contents

- [Workflow](#workflow)
- [build_engine.py](#build_enginepy)
- [Config Parameters](#config-parameters)
- [Quantization Modes](#quantization-modes)
- [configs/](#configs)
- [trt_checkpoints/](#trt_checkpoints)
- [trt_engines/](#trt_engines)
- [tokenizers/](#tokenizers)

---

## Workflow

1. **Quantize + checkpoint** — the model is loaded from HuggingFace, quantized according to the config, and saved as a TensorRT-LLM checkpoint to `trt_checkpoints/`
2. **Compile engine** — the checkpoint is compiled into a TensorRT engine and saved to `trt_engines/`

Both steps are handled by a single run of `build_engine.py`. If a pre-built checkpoint already exists, it can be provided via `checkpoint_in_dir` to skip the quantization step.

---

## build_engine.py

Main entry point. Reads a build config, runs quantization and calibration if needed, saves a checkpoint, then compiles and saves a TensorRT engine.

**Supported models:** `qwen`, `llama`, `mistral`

Usage (typically invoked via an HPC or Docker job script):

```bash
python build_engine.py \
  --config path/to/build_config.json \
  --base path/to/trt-scripts \
  --model-dir path/to/hf-model
```

---

## Config Parameters

Configs are JSON files passed to `build_engine.py`. All paths are relative to their respective root directories unless absolute.

### Model

| Parameter | Type | Description |
|---|---|---|
| `model_type` | string | Model architecture. Supported: `qwen`, `llama`, `mistral` |
| `dtype` | string | Base compute dtype. Typically `float16` |

### Quantization

| Parameter | Type | Description |
|---|---|---|
| `quant_mode` | string | Quantization scheme to apply. See [Quantization Modes](#quantization-modes) |
| `kv_cache_dtype` | string \| null | KV cache quantization dtype. Options: `fp8`, `int8`. `null` uses the model dtype |
| `quant_backend` | string | Quantization backend. Options: `auto`, `legacy`, `modelopt`. Defaults to `auto` when omitted. `auto` uses ModelOpt for AWQ/FP8 modes and FP8 KV-cache, and keeps RTN and SmoothQuant modes on the legacy converter so INT8 KV-cache scales are exported |

`quant_backend=modelopt` rejects `kv_cache_dtype: "int8"` because ModelOpt has been observed to export `kv_cache_quant_algo: null` for INT8 KV-cache. Use the legacy backend for RTN/SQ + INT8 KV-cache modes. `quant_backend=legacy` rejects FP8 KV-cache because the legacy converter only implements INT8 KV-cache scaling.

### Engine Build

| Parameter | Type | Description |
|---|---|---|
| `max_batch_size` | int | Maximum number of requests processed in parallel |
| `max_input_len` | int | Maximum number of input tokens per request |
| `max_seq_len` | int | Maximum total sequence length (input + output) per request |
| `max_num_tokens` | int | Maximum total tokens across all requests in a batch |
| `max_beam_width` | int | Beam search width. `1` = greedy decoding |
| `gather_context_logits` | bool | Output per-token logits over the full context. Required for loglikelihood-based eval tasks (e.g. MMLU, HellaSwag). Engines with this enabled use the `_LOGITS` suffix by convention. **Note:** enabling this comes with significant performance penalties — only use it for evaluation, not serving |

### Calibration

Used by quantization methods that require calibration data (e.g. `W8A8_SQ`, `W4A16_AWQ`, `W4A8_AWQ`). Ignored for `W16A16` and `W8A16`.

| Parameter | Type | Description |
|---|---|---|
| `calib_source` | string | HuggingFace dataset name or local path used for calibration |
| `calib_split` | string | Dataset split to use (e.g. `train`) |
| `calib_text_field` | string | Field in the dataset containing the text samples |
| `calib_num_samples` | int | Number of samples to use for calibration |
| `calib_batch_size` | int | Batch size during calibration |
| `calib_max_seq_length` | int | Maximum token length for calibration samples |
| `random_seed` | int | Seed for reproducibility |

### Paths

| Parameter | Type | Description |
|---|---|---|
| `engine_out_dir` | string | Output path for the compiled engine, relative to `model/trt_engines/` |
| `checkpoint_out_dir` | string \| null | Output path for the quantized checkpoint, relative to `model/trt_checkpoints/`. Can be `null` if building from an existing checkpoint via `checkpoint_in_dir` |
| `checkpoint_in_dir` | string \| null | Path to an existing checkpoint to build from, skipping quantization. Relative to `model/trt_checkpoints/` |

---

## Quantization Modes

| Mode | Weights | Activations | Notes |
|---|---|---|---|
| `W16A16` | float16 | float16 | No quantization. Baseline |
| `W8A16` | int8 | float16 | Weight-only quantization |
| `W4A16` | int4 | float16 | Weight-only quantization (RTN) |
| `W4A16_AWQ` | int4 | float16 | Weight-only with AWQ calibration |
| `W4A8_AWQ` | int4 | fp8 | AWQ weights + quantized activations. Requires Hopper architecture or newer |
| `W8A8_SQ` | int8 | int8 | SmoothQuant — requires calibration |
| `FP8` | fp8 | fp8 | Requires Hopper architecture or newer |

---

## `configs/`

Build configs organized by environment, model, and quantization variant:

```
configs/
  hpc/<model>/<quant>/
    build_config_<QUANT>_LOGITS.json   # conversion + LOGITS engine (batch 8)
  docker/<model>/
    build_existing_<QUANT>             # bench engine from checkpoint (batch 1, no logits)
```

For example:
- `configs/hpc/Qwen25_7B/W8A8_SQ/build_config_W8A8_SQ_LOGITS.json` — full conversion on HPC
- `configs/docker/Qwen25_7B/build_existing_W8A8_SQ` — bench engine rebuild from checkpoint

HPC configs have no `.json` extension convention difference from docker configs, which use the `build_existing_*` prefix and no `.json` extension.

---

## `trt_checkpoints/`

Quantized TensorRT-LLM checkpoints produced by the quantization step, organized by model and quantization variant. These can be reused via `checkpoint_in_dir` to skip re-quantization when rebuilding an engine.

```
trt_checkpoints/
  <model>/
    <QUANT>/
      config.json
      rank0.safetensors
```

---

## `trt_engines/`

Compiled TensorRT engines ready for inference, organized by model and quantization variant.

```
trt_engines/
  <model>/
    <QUANT>/
      config.json
      rank0.engine
```

The `_LOGITS` suffix in engine directory names indicates the engine was built with `gather_context_logits: true`, which is required for loglikelihood evaluation tasks.

---

## `tokenizers/`

HuggingFace tokenizer files for each model family, stored locally so bench jobs do not need a full model snapshot. Currently contains `Qwen25/` and `Mistral/`. The tokenizer path is passed to `bench_latency.py` via `--tokenizer`.

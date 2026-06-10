# trt-scripts

Build and evaluation scripts for a master's thesis benchmarking weight and KV-cache quantization for local LLM inference via TensorRT-LLM. The goal is reproducibility: a reader who gets different numbers should be able to trace exactly where this pipeline diverges from upstream TRT-LLM defaults. See the thesis for analysis and findings.

---

## Environment

**Container:** `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5`  
**OS:** Ubuntu 24.04 · CUDA 13.0 · driver 580.82.09

> **Warning:** The monkeypatches in this repo target internal APIs of TRT-LLM 1.1.0rc5. They will silently break or throw errors on other versions. Do not upgrade the container without auditing every patch site listed below.

---

## Pipeline Overview

The pipeline has two decoupled stages.

**Stage 1 — Checkpoint conversion** ([model/build_engine.py](model/build_engine.py)):  
HuggingFace model → TRT-LLM checkpoint, with optional quantization/calibration. Conversion runs once per (model, quant) pair on HPC. Checkpoints are stored in `model/trt_checkpoints/`.

**Stage 2 — Engine build** ([model/build_engine.py](model/build_engine.py)):  
TRT-LLM checkpoint → deployable TRT engine with fixed batch/sequence bounds. One checkpoint can produce multiple engines with different settings — e.g., a `_LOGITS` engine for eval and a leaner bench engine. Engines are GPU-architecture-specific and must be rebuilt per environment.

The script accepts either a `model_dir` (HF snapshot, runs both stages) or a `checkpoint_in_dir` (skips conversion, engine build only). The docker rebuild scripts (`docker/jobs/*/build_all_*_from_checkpoints.sh`) use the latter path.

---

## Build Config

Each (model, quantization) pair is described by a JSON config file under `model/configs/<env>/<model>/<quant>/`. All fields map directly to the `BuildConfig` dataclass in [model/build_engine.py](model/build_engine.py). Configs for the same (model, quant) differ between HPC and Docker only in `engine_out_dir`, `gather_context_logits`, and `max_batch_size`; all quant and calibration fields are identical.

### Path fields

| Field | Description |
|---|---|
| `engine_out_dir` | Engine output, relative to `model/trt_engines/` or absolute |
| `checkpoint_out_dir` | Checkpoint output, relative to `model/trt_checkpoints/` or absolute |
| `checkpoint_in_dir` | Load a pre-built checkpoint instead of converting. Mutually exclusive with `model_dir`. |

### Model fields

| Field | Default | Description |
|---|---|---|
| `model_type` | `"qwen"` | `"qwen"`, `"llama"`, or `"mistral"` — selects `QWenForCausalLM` or `LLaMAForCausalLM` |
| `dtype` | `"float16"` | Base compute dtype; `"bfloat16"` in all thesis configs |

### Quantization fields

| Field | Default | Description |
|---|---|---|
| `quant_mode` | `"W4A16"` | Weight/activation scheme: `W16A16`, `W8A16`, `W4A16`, `W4A16_AWQ`, `W4A8_AWQ`, `W8A8_SQ`, `FP8` |
| `kv_cache_dtype` | `null` | KV-cache precision: `"int8"`, `"fp8"`, or `null` |
| `quant_backend` | `"auto"` | Conversion path selector: `"auto"`, `"legacy"`, `"modelopt"`. See [Conversion Paths](#conversion-paths). |

### Engine build fields

| Field | Default | Description |
|---|---|---|
| `max_batch_size` | `32` | Max concurrent requests. HPC configs use `8`; Docker bench configs use `1`. |
| `max_input_len` | `2048` | Max prompt length in tokens |
| `max_seq_len` | `6144` | Max total sequence length (prompt + output) |
| `max_num_tokens` | `12288` | Max tokens across all in-flight requests |
| `max_beam_width` | `1` | Beam search width (1 = greedy) |
| `gather_context_logits` | `false` | Must be `true` for loglikelihood eval tasks (MMLU, HellaSwag, etc.). Increases engine memory footprint significantly. |

Engines built with `gather_context_logits: true` are suffixed `_LOGITS` in directory names. Bench engines are rebuilt without logits to reclaim VRAM.

### Calibration fields

| Field | Default | Description |
|---|---|---|
| `calib_source` | `"neuralmagic/LLM_compression_calibration"` | HuggingFace dataset identifier |
| `calib_split` | `"train"` | Dataset split |
| `calib_text_field` | `"text"` | Column used as calibration text |
| `calib_num_samples` | `2048` | Target sample count. **See caveat below.** |
| `calib_batch_size` | `16` | Tokenizer batch size during calibration |
| `calib_max_seq_length` | `6144` | Truncation length for calibration sequences |
| `random_seed` | `0` | Seed for calibration sampling |

> **Calibration sample count caveat (upstream behavior):** `calib_num_samples` is respected on the ModelOpt path (Path A). On the legacy paths (Path B/C), the underlying TRT-LLM converters impose their own limits: the LLaMA/Mistral path caps at 128 samples (divides internally by batch size); the Qwen path hardcodes 512 samples and appends `" TL;DR: "` to each prompt. This is TRT-LLM 1.1.0rc5 behavior, not a bug in these scripts. It affects calibration dataset effective size for those variants.

---

## Conversion Paths

`build_engine.py:convert_and_quantize()` chooses among three paths. Dispatch is driven by the `quant_backend` value resolved by `_resolve_quant_backend()` and by whether INT8 KV-cache or SmoothQuant calibration is required.

### Path A — ModelOpt with calibration

**Trigger:** `quant_backend` resolves to `"modelopt"`  
**Applies to:** `W4A16_AWQ`, `W4A8_AWQ`, `FP8`, and any mode combined with `kv_cache_dtype=fp8`

Calls `ModelClass.quantize()` via the ModelOpt backend. The calibration dataloader is replaced by [Patch 1](#patch-1--calibration-dataloader). Produces a checkpoint that is then loaded with `from_checkpoint()`.

### Path B — Legacy converter with calibration

**Trigger:** `quant_backend` resolves to `"legacy"` AND (`kv_cache_dtype == "int8"` OR `quant_mode` is `W8A8_SQ`)  
**Applies to:** `W16A16_INT8KV`, `W8A16_INT8KV`, `W4A16_INT8KV`, `W8A8_SQ`, `W8A8_SQ_INT8KV`

Calls `ModelClass.quantize()` via the legacy converter (takes `dtype` and `mapping` arguments, unlike Path A). This path is required because the ModelOpt backend does not implement INT8 KV-cache scaling on Ampere hardware.

### Path C — Legacy converter without calibration

**Trigger:** `quant_backend` resolves to `"legacy"` AND no calibration needed  
**Applies to:** `W16A16` (BF16 baseline), `W8A16`, `W4A16` (RTN)

Calls `ModelClass.from_hugging_face()` directly — no quantization/calibration pass.

### The `quant_backend` field

`quant_backend` was added after discovering that `W4A16_AWQ + kv_cache_dtype=int8` silently produced a plain AWQ engine — the ModelOpt backend exported `kv_cache_quant_algo: null` and discarded the INT8 KV setting without raising any error. The explicit field makes the dispatch auditable and fails loudly on unsupported combinations.

`"auto"` resolves as follows:

| Condition | Resolved backend |
|---|---|
| `kv_cache_dtype=fp8` | `modelopt` |
| `kv_cache_dtype=int8` with a legacy-compatible weight mode | `legacy` |
| AWQ or FP8 weight modes | `modelopt` |
| Everything else | `legacy` |

Hard errors:
- `quant_backend=modelopt` + `kv_cache_dtype=int8` → raises (the known silent-failure case)
- `quant_backend=legacy` + `kv_cache_dtype=fp8` → raises (unsupported combination)

---

## Monkeypatches

The container image cannot be edited in place. All patches are applied at script-import time by direct attribute reassignment. They do not change quantization logic; they expose configuration surface that upstream hardcodes or leaves unimplemented.

### Patch 1 — Calibration dataloader

**File:** [model/build_engine.py:26–61](model/build_engine.py)  
**Target:** `tensorrt_llm.quantization.quantize_by_modelopt.get_calib_dataloader`

Upstream hardcodes `cnn_dailymail` as the calibration dataset. The patched version accepts `dataset_name_or_dir`, `split`, and `text_field` arguments, allowing any HF dataset to be configured. This thesis uses `neuralmagic/LLM_compression_calibration`.

### Patch 2 — Bench `model_type` property

**File:** [bench/bench_latency.py:92–111](bench/bench_latency.py)  
**Target:** `tensorrt_llm.bench.benchmark.GeneralExecSettings.model_type`

The upstream property calls `AutoConfig.from_pretrained`, but only tokenizer files (not full model configs) are stored locally. The patched property returns `None` when `self.modality is None`, short-circuiting the HF config load entirely.

### Patch 3 — Bench sampler options

**File:** [bench/bench_latency.py:92–111](bench/bench_latency.py)  
**Target:** `tensorrt_llm.bench.benchmark.low_latency.update_sampler_args_with_extra_options`

Upstream requires a separate YAML file for sampler settings. The patch replaces that requirement with the `sampler_options` dict from `BenchConfig`, configured via `bench/configs/base_config.json`.

### Patch 4 — `generate_until`

**File:** [eval/lmeval_patches.py:8–31](eval/lmeval_patches.py)  
**Target:** `tensorrt_llm.evaluate.lm_eval.LmEvalWrapper.generate_until`

Wraps the generation loop with profiling and optional debug output (controlled by `print_outputs` in the eval config).

### Patch 5 — `_loglikelihood_tokens`

**File:** [eval/lmeval_patches.py:34–65](eval/lmeval_patches.py)  
**Target:** `tensorrt_llm.evaluate.lm_eval.LmEvalWrapper._loglikelihood_tokens`

Upstream marks this method as not implemented. The patch implements it using `context_logits` from the engine output — required for all multiple-choice accuracy tasks (MMLU, HellaSwag, WinoGrande, GPQA). Engine must be built with `gather_context_logits: true`.

### Patch 6 — `loglikelihood_rolling`

**File:** [eval/lmeval_patches.py:69–102](eval/lmeval_patches.py)  
**Target:** `tensorrt_llm.evaluate.lm_eval.LmEvalWrapper.loglikelihood_rolling`

Upstream marks this method as not implemented. The patch implements rolling-window perplexity evaluation required for the WikiText task, delegating to the patched `_loglikelihood_tokens` above.

### Patch 7 — `confirm_run_unsafe_code`

**File:** [eval/lmeval_patches.py:124–128](eval/lmeval_patches.py)  
**Target:** `lm_eval.evaluator.evaluate`

HumanEval and MBPP require executing generated code. The patch wraps `evaluate` to always pass `confirm_run_unsafe_code=True`, removing the interactive confirmation prompt that would block unattended evaluation.

---

## Execution Environments

### HPC (A100, Apptainer, SLURM)

Used for all checkpoint conversions and for building LOGITS engines.

- Container: Apptainer SIF built from [hpc/trtllm-tools.def](hpc/trtllm-tools.def)
- Jobs: SLURM array scripts in `hpc/jobs/<model>/<quant>/`
- Engine params: `max_batch_size=8`, `gather_context_logits=true`
- Eval: `free_gpu_memory_fraction=0.4` (KV headroom for context-logit passes), tasks dispatched by `SLURM_ARRAY_TASK_ID` over 8 task configs

### Local (RTX 3090, Docker)

Used for rebuilding bench engines (no logits) and running latency benchmarks.

- Container: Docker image from [docker/Dockerfile](docker/Dockerfile) — same base image as Apptainer
- Jobs: serial bash loops in `docker/jobs/<model>/`
- Engine params: `max_batch_size=1`, `gather_context_logits=false`, `kv_cache_free_gpu_mem_fraction=0.9`

---

## Running the Pipeline

### Build a checkpoint and engine (HPC, full conversion)

```bash
python model/build_engine.py \
  --config model/configs/hpc/Qwen25_7B/W4A16_AWQ/build_config_W4A16_AWQ_LOGITS.json \
  --base /path/to/trt-scripts \
  --model-dir /path/to/hf-cache/Qwen2.5-7B
```

### Build a bench engine from an existing checkpoint (Docker)

```bash
# All variants for a model:
bash docker/jobs/Qwen25_7B/build_all_qwen25_7b_from_checkpoints.sh

# Single variant:
python model/build_engine.py \
  --config model/configs/docker/Qwen25_7B/build_existing_W4A16_AWQ \
  --base /path/to/trt-scripts
```

### Run latency benchmarks (Docker)

```bash
# All variants and datasets for a model:
bash docker/jobs/Qwen25_7B/bench_all_qwen25_7b.sh

# Single run:
python bench/bench_latency.py \
  --base /path/to/trt-scripts \
  --engine Qwen25_7B/W4A16_AWQ \
  --dataset Qwen_Synthetic_short_in_short_out_256_256_16_16.txt \
  --tokenizer Qwen25
```

Results are appended to `bench/results/results.csv`.

### Run quality evaluation (HPC)

```bash
# All 8 tasks as a SLURM array:
sbatch --array=0-7 hpc/jobs/Qwen25_7B/W4A16_AWQ/eval_array_W4A16_AWQ.sh

# Single task:
python eval/custom_lmeval_wrapper.py \
  --config eval/configs/tasks/eval_mmlu.json \
  --base /path/to/trt-scripts \
  --engine-dir Qwen25_7B/W4A16_AWQ_LOGITS \
  --model-dir /path/to/hf-cache/Qwen2.5-7B
```

Results are appended to `results.csv` in the working directory.

---

## Evaluation Tasks

| Task | Config | Mode | Notes |
|---|---|---|---|
| HumanEval | `eval_humaneval.json` | Generation | Requires `HF_ALLOW_CODE_EVAL=1` (set automatically) |
| MBPP | `eval_mbpp.json` | Generation | Requires `HF_ALLOW_CODE_EVAL=1` (set automatically) |
| MMLU | `eval_mmlu.json` | Loglikelihood | Requires LOGITS engine |
| HellaSwag | `eval_hellaswag.json` | Loglikelihood | Requires LOGITS engine |
| WinoGrande | `eval_winogrande.json` | Loglikelihood | Requires LOGITS engine |
| GPQA | `eval_gpqa.json` | Loglikelihood | Requires LOGITS engine; gated dataset (`HF_TOKEN` read from `.hf_token`) |
| GSM8K | `eval_gsm8k.json` | Generation | — |
| WikiText | `eval_wikitext.json` | Perplexity | Rolling loglikelihood; `LmEvalEvaluator` result divided by 100 post-eval |

---

## Repo Layout

```
trt-scripts/
├── model/
│   ├── build_engine.py              # Conversion + engine build (all three paths)
│   ├── extract_scales.py            # Inspect quantization scales from a checkpoint
│   ├── configs/
│   │   ├── hpc/<model>/<quant>/     # LOGITS engines, batch 8, gather_context_logits=true
│   │   └── docker/<model>/<quant>/  # Bench engines, batch 1, gather_context_logits=false
│   ├── trt_checkpoints/             # Saved quantized checkpoints (not in git)
│   ├── trt_engines/                 # Compiled TRT engines (not in git)
│   └── tokenizers/                  # HF tokenizer files (not in git)
├── bench/
│   ├── bench_latency.py             # Latency benchmark wrapper (wraps trtllm-bench)
│   ├── configs/base_config.json     # Default bench settings
│   ├── datasets/                    # Synthetic prompt datasets + generation scripts
│   └── results/                     # Benchmark output CSVs
├── eval/
│   ├── custom_lmeval_wrapper.py     # lm-eval harness wrapper
│   ├── lmeval_patches.py            # LmEvalWrapper and evaluator patches
│   └── configs/tasks/               # Per-task eval configs (8 tasks)
├── hpc/
│   ├── trtllm-tools.def             # Apptainer container definition
│   └── jobs/<model>/<quant>/        # SLURM array job scripts
└── docker/
    ├── Dockerfile                   # Docker image (same base as Apptainer)
    ├── Makefile                     # make build / run-shell / run-jupyter
    └── jobs/<model>/                # Serial build + bench scripts
```

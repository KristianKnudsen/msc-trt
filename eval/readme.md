# eval/

Contains the evaluation pipeline for running benchmarks against TensorRT-LLM engines using `lm-eval`.

### Contents

- [custom_lmeval_wrapper.py](#custom_lmeval_wrapperpy)
- [Config Parameters](#config-parameters)
- [Supported Tasks](#supported-tasks)
- [Output](#output)
- [configs/](#configs)

---

## custom_lmeval_wrapper.py

Wraps TensorRT-LLM's `LmEvalWrapper` to run `lm-eval` benchmarks against a compiled engine. Handles task setup, engine initialisation, evaluation, and result logging.

**Notable behaviour:**

- **Loglikelihood tasks** (e.g. MMLU, HellaSwag) use a custom implementation that processes requests one at a time rather than in batches. This works around a memory accumulation issue in the TRT-LLM engine when `gather_context_logits` is enabled over large batches.
- **Code eval tasks** (`humaneval`, `mbpp`) automatically set `HF_ALLOW_CODE_EVAL=1` to permit execution of generated code.
- **GPQA** requires a HuggingFace token due to the dataset being gated. The wrapper reads it automatically from `.hf_token` in the repo root.

Usage (typically invoked via an HPC or Docker job script):

```bash
python eval/custom_lmeval_wrapper.py \
  --config path/to/eval_config.json \
  --base path/to/trt-scripts \
  --engine-dir Qwen25_3B/W16A16_LOGITS \
  --model-dir path/to/hf-model
```

---

## Config Parameters

### Engine

| Parameter | Type | Description |
|---|---|---|
| `free_gpu_memory_fraction` | float | Fraction of available GPU memory to reserve for the KV cache |
| `max_batch_size` | int | Maximum number of requests processed in parallel. Set to `1` for loglikelihood tasks |
| `max_tokens` | int | Maximum number of output tokens per request. Set to `1` for loglikelihood tasks |

### Sampling

| Parameter | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature. `0.0` = greedy decoding |
| `return_context_logits` | bool | Return per-token logits over the full context. Required for loglikelihood-based tasks. Must also be enabled in the engine (`gather_context_logits`) |
| `seed` | int | Random seed for reproducibility |

### Evaluation

| Parameter | Type | Description |
|---|---|---|
| `task` | string | `lm-eval` task name (e.g. `mmlu`, `humaneval`, `gpqa_main_zeroshot`) |
| `few_shots` | int | Number of few-shot examples to prepend |
| `num_samples` | int \| null | Number of dataset samples to evaluate. `null` = full dataset |
| `scores_filter` | string \| null | `metric,aggregation` pair to extract from results (e.g. `acc,none`). `null` = all metrics |
| `print_outputs` | int | Number of input/output pairs to print during evaluation for debugging. `0` = silent |

### Paths (CLI arguments)

Engine and model paths are not stored in the task configs — they are passed as CLI arguments so the same task config works across all models and quant variants:

| Argument | Description |
|---|---|
| `--engine-dir` | Path to the compiled TRT engine, absolute or relative to `model/trt_engines/` |
| `--model-dir` | Path to the HuggingFace model snapshot (used as tokenizer source) |
| `--base` | Path to the `trt-scripts` root directory |

`engine_dir` can also be set in the config JSON as a fallback, but the CLI argument takes priority.

---

## Supported Tasks

| Task | `task` value | Few-shots | Mode | Notes |
|---|---|---|---|---|
| HumanEval | `humaneval` | 0 | Generation | Requires code execution (`HF_ALLOW_CODE_EVAL=1` set automatically) |
| MBPP | `mbpp` | 3 | Generation | Requires code execution (`HF_ALLOW_CODE_EVAL=1` set automatically) |
| MMLU | `mmlu` | 5 | Loglikelihood | Requires LOGITS engine |
| MMLU-Pro | `mmlu_pro` | 5 | Generation (1 token) | Answer-letter generation; does not require LOGITS engine |
| HellaSwag | `hellaswag` | 5 | Loglikelihood | Requires LOGITS engine |
| WinoGrande | `winogrande` | 5 | Loglikelihood | Requires LOGITS engine |
| GPQA | `gpqa_main_zeroshot` | 0 | Loglikelihood | Requires LOGITS engine; gated dataset (`HF_TOKEN` read from `.hf_token`) |
| GSM8K | `gsm8k` | 5 | Generation | — |
| WikiText | `wikitext` | 0 | Perplexity | Requires LOGITS engine; rolling loglikelihood; result divided by 100 post-eval |

---

## Output

Results are appended to `results.csv` in the working directory after each run. Columns:

| Column | Description |
|---|---|
| `model` | Engine directory name |
| `task` | Task name |
| `result` | Accuracy or score (rounded to 2 decimal places) |
| `n_samples_actual` | Number of samples actually evaluated (deduped for loglikelihood tasks) |
| *(EvalConfig fields)* | All `EvalConfig` fields except `task` (free_gpu_memory_fraction, max_batch_size, etc.) |
| `engine_disk_mib` | Engine `.engine` file size on disk (MiB) |
| `exec_context_mib` | GPU memory for execution context (MiB) |
| `kv_cache_alloc_mib` | GPU memory allocated for KV cache (MiB) |
| `runtime_buffers_mib` | GPU memory for runtime buffers (MiB) |
| `decoder_mib` | GPU memory for the decoder (MiB) |

---

## `configs/`

Eval configs are shared across models and quantization variants. The engine is selected at runtime with `--engine-dir`:

```
configs/
  tasks/
    eval_<task>.json
```

For example:

```bash
python eval/custom_lmeval_wrapper.py \
  --config eval/configs/tasks/eval_mmlu.json \
  --base /path/to/trt-scripts \
  --model-dir /path/to/hf-model \
  --engine-dir Qwen25_3B/W8A8_SQ_LOGITS
```

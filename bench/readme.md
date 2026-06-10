# bench/

Latency benchmarking for compiled TensorRT engines. Wraps `trtllm-bench latency` with nvidia-smi monitoring and appends one row per run to `results/results.csv`.

### Contents

- [bench_latency.py](#bench_latencypy)
- [Config Parameters](#config-parameters)
- [Datasets](#datasets)
- [Output](#output)

---

## bench_latency.py

Wrapper around `trtllm-bench latency` (`tensorrt_llm.commands.bench`). For each run it:

1. Starts nvidia-smi polling at a configurable interval
2. Calls `trtllm-bench latency` with the given engine, dataset, and config
3. Stops smi, parses peak GPU utilisation/power/memory from the poll CSV
4. Extracts latency percentiles and throughput metrics from the report JSON
5. Appends one row to `results/results.csv`

Monkeypatches applied at import time (see root README for details):
- `GeneralExecSettings.model_type` — returns `None` when `modality` is unset, avoiding `AutoConfig.from_pretrained` when only tokenizer files are present locally
- `update_sampler_args_with_extra_options` — uses `sampler_options` dict from config instead of requiring a YAML file

Usage (typically invoked via a Docker job script):

```bash
python bench/bench_latency.py \
  --base /path/to/trt-scripts \
  --engine Qwen25_7B/W4A16_AWQ \
  --dataset Qwen_Synthetic_short_in_short_out_256_256_16_16.txt \
  --tokenizer Qwen25
```

| Argument | Default | Description |
|---|---|---|
| `--base` | `/workspace/trt-scripts` | Repo root path |
| `--engine` | *(required)* | Engine path under `model/trt_engines/` (e.g. `Qwen25_7B/W4A16_AWQ`) |
| `--dataset` | *(required)* | Dataset filename under `bench/datasets/` |
| `--config` | `base_config.json` | Config filename under `bench/configs/`. The `.json` suffix is optional. |
| `--tokenizer` | *(required)* | Tokenizer directory under `model/tokenizers/` (e.g. `Qwen25`) |

---

## Config Parameters

Config files live under `bench/configs/`. The default is [`base_config.json`](configs/base_config.json).

| Parameter | Default | Description |
|---|---|---|
| `num_requests` | `10` | Number of requests per run |
| `warmup` | `1` | Warmup requests before measurement (not included in results) |
| `backend` | `"tensorrt"` | trtllm-bench backend identifier |
| `kv_cache_free_gpu_mem_fraction` | `0.9` | Fraction of GPU memory to allocate for the KV cache |
| `concurrency` | `1` | Number of concurrent in-flight requests |
| `max_batch_size` | `1` | Max requests per batch |
| `smi_logging` | `true` | Enable nvidia-smi polling during the run |
| `smi_interval_ms` | `1000` | nvidia-smi polling interval in milliseconds |
| `keep_raw` | `false` | Keep per-run `report.json`, `smi.csv`, and workspace after the run |
| `sampler_options` | see below | Sampling parameters injected into trtllm-bench via the patch |

Default `sampler_options`:
```json
{
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0
}
```

---

## Datasets

Synthetic prompt datasets live under `bench/datasets/`. Each file is a pre-generated list of token sequences sampled from a normal distribution over input/output length, produced by `trtllm-bench prepare_dataset`.

**Naming convention:** `<Tokenizer>_Synthetic_<profile>_<input_mean>_<output_mean>_<input_stdev>_<output_stdev>.txt`

Four profiles are benchmarked per model:

| Profile | Input mean (tokens) | Output mean (tokens) |
|---|---|---|
| `short_in_short_out` | 256 | 256 |
| `long_in_short_out` | 2560 | 256 |
| `short_in_long_out` | 256 | 2560 |
| `long_in_long_out` | 2560 | 2560 |

Regenerate with the scripts in `bench/datasets/` (paths inside the scripts assume the container workspace layout):

```bash
bash bench/datasets/GenerateQwenData.sh
bash bench/datasets/GenerateMistralData.sh
```

---

## Output

Results are appended to `bench/results/results.csv`. One row per (engine, dataset) combination.

Raw per-run files (`report.json`, `smi.csv`, `iteration.log`) are written to `bench/results/raw/` and deleted after the row is written unless `keep_raw: true`.

### Run metadata

| Column | Description |
|---|---|
| `timestamp` | Run start time |
| `model` | First path component of `--engine` (e.g. `Qwen25_7B`) |
| `quant` | Remaining path components of `--engine` (e.g. `W4A16_AWQ`) |
| `engine` | Full `--engine` argument |
| `dataset` | Dataset filename |
| `config` | Config filename |
| `num_requests` | Number of requests in the run |

### Engine info (from trtllm-bench report)

| Column | Description |
|---|---|
| `backend` | Inference backend |
| `dtype` | Engine compute dtype |
| `quantization` | Weight quantization scheme |
| `kv_cache_dtype` | KV cache dtype |
| `max_input_length` | Engine max input length |
| `max_sequence_length` | Engine max sequence length |
| `tp_size` | Tensor parallelism degree |
| `pp_size` | Pipeline parallelism degree |

### Dataset stats

| Column | Description |
|---|---|
| `isl_average` | Average input sequence length across the run |
| `osl_average` | Average output sequence length |
| `seq_average` | Average total sequence length |

### Latency and throughput

| Column | Description |
|---|---|
| `total_latency_ms` | Total wall-clock time for all requests |
| `avg_request_latency_ms` | Mean per-request latency |
| `request_throughput_req_s` | Requests per second |
| `system_output_throughput_tok_s` | Output tokens per second (system-level) |
| `system_total_throughput_tok_s` | Total tokens per second (input + output) |
| `token_output_speed_tok_s` | Per-request output token speed |
| `avg_ttft_ms` | Average time-to-first-token |
| `avg_tpot_ms` | Average time-per-output-token |
| `latency_ms_{average,p50,p90,p95,p99}` | Request latency percentiles |
| `ttft_ms_{average,p50,p90,p95,p99}` | TTFT percentiles |
| `tpot_ms_{average,p50,p90,p95,p99}` | TPOT percentiles |
| `gen_tps_{average,p50,p90,p95,p99}` | Generation tokens/s percentiles |

### GPU stats (from nvidia-smi)

| Column | Description |
|---|---|
| `gpu_util_peak_pct` | Peak GPU compute utilisation (%) |
| `gpu_memory_util_peak_pct` | Peak GPU memory utilisation (%) |
| `gpu_memory_used_peak_mb` | Peak GPU memory used (MB) |
| `gpu_power_peak_w` | Peak GPU power draw (W) |

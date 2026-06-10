#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


SMI_QUERY = (
    "timestamp,name,utilization.gpu,utilization.memory,"
    "memory.total,memory.used,memory.free,power.draw"
)

PERCENTILES = ("average", "p50", "p90", "p95", "p99")

RESULT_COLUMNS = [
    "timestamp",
    "model",
    "quant",
    "engine",
    "dataset",
    "config",
    "num_requests",
    "backend",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "max_input_length",
    "max_sequence_length",
    "tp_size",
    "pp_size",
    "isl_average",
    "osl_average",
    "seq_average",
    "total_latency_ms",
    "avg_request_latency_ms",
    "request_throughput_req_s",
    "system_output_throughput_tok_s",
    "system_total_throughput_tok_s",
    "token_output_speed_tok_s",
    "avg_ttft_ms",
    "avg_tpot_ms",
    "latency_ms_average",
    "latency_ms_p50",
    "latency_ms_p90",
    "latency_ms_p95",
    "latency_ms_p99",
    "ttft_ms_average",
    "ttft_ms_p50",
    "ttft_ms_p90",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "tpot_ms_average",
    "tpot_ms_p50",
    "tpot_ms_p90",
    "tpot_ms_p95",
    "tpot_ms_p99",
    "gen_tps_average",
    "gen_tps_p50",
    "gen_tps_p90",
    "gen_tps_p95",
    "gen_tps_p99",
    "gpu_util_peak_pct",
    "gpu_memory_util_peak_pct",
    "gpu_memory_used_peak_mb",
    "gpu_power_peak_w",
]


@dataclass
class BenchConfig:
    num_requests: int = 10
    warmup: int = 1
    backend: str = "tensorrt"
    kv_cache_free_gpu_mem_fraction: float = 0.9
    concurrency: int = 1
    max_batch_size: int = 1
    keep_raw: bool = False
    smi_logging: bool = True
    smi_interval_ms: int = 1000
    sampler_options: dict[str, float] = field(default_factory=lambda: {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    })


def patch_trtllm_bench(config: BenchConfig) -> None:
    from tensorrt_llm.bench.benchmark import GeneralExecSettings
    import tensorrt_llm.bench.benchmark.low_latency as low_latency

    original_model_type = GeneralExecSettings.model_type
    original_sampler_update = low_latency.update_sampler_args_with_extra_options

    def model_type(self):
        if self.modality is None:
            return None
        return original_model_type.fget(self)

    # This patch removes the need for a sampler yaml file, allowing us to just send a di
    def update_sampler_args(sampler_args, sampler_options_path):
        if sampler_options_path is not None:
            return original_sampler_update(sampler_args, sampler_options_path)
        return sampler_args | config.sampler_options

    GeneralExecSettings.model_type = property(model_type)
    low_latency.update_sampler_args_with_extra_options = update_sampler_args


def load_config(path: Path) -> BenchConfig:
    with path.open() as f:
        data = json.load(f)
    return BenchConfig(**data)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_config(base: Path, config: str) -> Path:
    path = resolve_path(base / "bench" / "configs", config)
    if path.suffix == "":
        path = path.with_suffix(".json")
    return path


def start_smi(path: Path, interval_ms: int):
    output = path.open("w")
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={SMI_QUERY}",
            "--format=csv",
            "-lms",
            str(interval_ms),
        ],
        stdout=output,
        stderr=subprocess.DEVNULL,
    )
    return proc, output


def stop_smi(proc, output) -> None:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if output is not None:
        output.close()


def parse_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", value)
    return float(match.group()) if match else None


def parse_smi_peaks(path: Path) -> dict:
    peaks = {
        "gpu_util_peak_pct": None,
        "gpu_memory_util_peak_pct": None,
        "gpu_memory_used_peak_mb": None,
        "gpu_power_peak_w": None,
    }
    if not path.exists():
        return peaks

    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 8:
                continue
            values = {
                "gpu_util_peak_pct": parse_number(row[2]),
                "gpu_memory_util_peak_pct": parse_number(row[3]),
                "gpu_memory_used_peak_mb": parse_number(row[5]),
                "gpu_power_peak_w": parse_number(row[7]),
            }
            for key, value in values.items():
                if value is not None:
                    peaks[key] = value if peaks[key] is None else max(peaks[key], value)
    return peaks


def percentile_columns(data: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": data.get(key) for key in PERCENTILES}


def stat_average(data: dict, stat_name: str) -> float | None:
    stats = data.get(stat_name, {})
    return stats.get("average")


def extract_result_row(
    report_path: Path,
    smi_path: Path,
    timestamp: str,
    model: str,
    quant: str,
    engine: str,
    dataset: str,
    config_name: str,
    num_requests: int,
) -> dict:
    with report_path.open() as f:
        report = json.load(f)

    engine_info = report.get("engine", {})
    world_info = report.get("world_info", {})
    performance = report.get("performance", {})
    streaming = report.get("streaming_metrics", {})
    dataset_info = report.get("dataset", {})

    row = {
        "timestamp": timestamp,
        "model": model,
        "quant": quant,
        "engine": engine,
        "dataset": dataset,
        "config": config_name,
        "num_requests": num_requests,
        "backend": engine_info.get("backend"),
        "dtype": engine_info.get("dtype"),
        "quantization": engine_info.get("quantization"),
        "kv_cache_dtype": engine_info.get("kv_cache_dtype"),
        "max_input_length": engine_info.get("max_input_length"),
        "max_sequence_length": engine_info.get("max_sequence_length"),
        "tp_size": world_info.get("tp_size"),
        "pp_size": world_info.get("pp_size"),
        "isl_average": dataset_info.get("avg_isl", stat_average(dataset_info, "isl_stats")),
        "osl_average": dataset_info.get("avg_osl", stat_average(dataset_info, "osl_stats")),
        "seq_average": dataset_info.get("avg_sequence_length", stat_average(dataset_info, "seq_len_stats")),
        "total_latency_ms": performance.get("total_latency_ms"),
        "avg_request_latency_ms": performance.get("avg_request_latency_ms"),
        "request_throughput_req_s": performance.get("request_throughput_req_s"),
        "system_output_throughput_tok_s": performance.get("system_output_throughput_tok_s"),
        "system_total_throughput_tok_s": performance.get("system_total_throughput_tok_s"),
        "token_output_speed_tok_s": streaming.get("token_output_speed_tok_s"),
        "avg_ttft_ms": streaming.get("avg_ttft_ms"),
        "avg_tpot_ms": streaming.get("avg_tpot_ms"),
    }
    row |= percentile_columns(performance.get("request_latency_percentiles_ms", {}), "latency_ms")
    row |= percentile_columns(streaming.get("ttft_percentiles", {}), "ttft_ms")
    row |= percentile_columns(streaming.get("tpot_percentiles", {}), "tpot_ms")
    row |= percentile_columns(streaming.get("gen_tps_percentiles", {}), "gen_tps")
    row |= parse_smi_peaks(smi_path)
    return row


def append_result_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_trtllm_bench(args: list[str]) -> None:
    from tensorrt_llm.commands.bench import main as trtllm_bench_main

    trtllm_bench_main.main(args=args, prog_name="trtllm-bench", standalone_mode=False)


def run_benchmarks(args) -> int:
    base = args.base.resolve()
    engine_parts = Path(args.engine).parts
    model = engine_parts[0]
    quant = "/".join(engine_parts[1:])
    engine_label = "_".join(engine_parts)
    engine_dir = resolve_path(base / "model" / "trt_engines", args.engine)
    dataset_path = resolve_path(base / "bench" / "datasets", args.dataset)
    config_path = resolve_config(base, args.config)
    tokenizer_dir = resolve_path(base / "model" / "tokenizers", args.tokenizer)
    config = load_config(config_path)
    patch_trtllm_bench(config)

    dataset_name = dataset_path.with_suffix("").relative_to(base / "bench" / "datasets")
    dataset_name = str(dataset_name).replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base / "bench" / "results"
    raw_dir = results_dir / "raw" / f"{engine_label}_{dataset_name}_{timestamp}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    report_path = raw_dir / "report.json"
    iteration_log = raw_dir / "iteration.log"
    smi_path = raw_dir / "smi.csv"
    smi_proc = None
    smi_output = None

    if config.smi_logging:
        smi_proc, smi_output = start_smi(smi_path, config.smi_interval_ms)

    try:
        run_trtllm_bench([
            "-m", model,
            "--model_path", str(tokenizer_dir),
            "--workspace", str(raw_dir / "workspace"),
            "latency",
            "--engine_dir", str(engine_dir),
            "--backend", config.backend,
            "--dataset", str(dataset_path),
            "--num_requests", str(config.num_requests),
            "--warmup", str(config.warmup),
            "--concurrency", str(config.concurrency),
            "--kv_cache_free_gpu_mem_fraction", str(config.kv_cache_free_gpu_mem_fraction),
            "--report_json", str(report_path),
            "--iteration_log", str(iteration_log),
        ])
    finally:
        stop_smi(smi_proc, smi_output)

    row = extract_result_row(
        report_path=report_path,
        smi_path=smi_path,
        timestamp=timestamp,
        model=model,
        quant=quant,
        engine=args.engine,
        dataset=args.dataset,
        config_name=args.config,
        num_requests=config.num_requests,
    )
    results_csv = results_dir / "results.csv"
    append_result_row(results_csv, row)

    if not config.keep_raw:
        shutil.rmtree(raw_dir)

    print(results_csv)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TensorRT-LLM latency benchmarks.")
    parser.add_argument("--base", type=Path, default=Path("/workspace/trt-scripts"))
    parser.add_argument("--engine", type=str, required=True, help="Engine path under model/trt_engines")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path under bench/datasets")
    parser.add_argument("--config", type=str, default="base_config.json", help="Config path under bench/configs")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer path under model/tokenizers")
    return run_benchmarks(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

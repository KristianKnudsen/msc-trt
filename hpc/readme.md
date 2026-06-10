## HPC

This directory contains everything needed to run build and evaluation jobs on the HPC cluster using SLURM. Jobs are containerised via Apptainer and submitted as batch jobs.

### Contents

- [Container](#container-trtllm-toolssif)
- [Engine Builds](#engine-builds)
- [Evaluation](#evaluation)

---

## Container (`trtllm-tools.sif`)

The Apptainer image (`trtllm-tools.sif`) is built from `trtllm-tools.def` and provides the runtime environment for all build and eval jobs. It is based on `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5`.

### Build

```bash
cd /localscratch
apptainer build --fakeroot trtllm-tools.sif ~/trt-scripts/hpc/trtllm-tools.def
mv trtllm-tools.sif ~/trt-scripts/hpc/
```

> `/localscratch` is used due to limitations on network-mounted home directories. Rebuild by rerunning after updating `.def`.

---

## Engine Builds

Build jobs invoke [`model/build_engine.py`](../model/build_engine.py) inside the container, which compiles a TensorRT-LLM engine for a given model and quantization. Refer to the [`model/`](../model) directory for build configs and documentation.

### Single build job

For building one engine at a time, use [`hpc/jobs/build_job_template.sh`](jobs/build_job_template.sh) as a starting point:

1. Copy the template into `hpc/jobs/<model>/`
2. Fill in the placeholders at the top of the script:
   - `<BASE_PATH>` — path to the `trt-scripts` root
   - `<MODEL>` — model subfolder name used in config paths (e.g. `qwen`)
   - `<MODEL_DIR>` — HuggingFace model directory name (e.g. `Qwen2.5-3B`)
   - `<QUANT>` — quantization name (e.g. `W16A16`)
3. Submit with:

```bash
sbatch hpc/jobs/<model>/your_build_job.sh
```

### Building multiple quants at once (array job)

To build several quantization variants in parallel, use [`hpc/jobs/build_array_job_template.sh`](jobs/build_array_job_template.sh). This submits one SLURM task per config, all running simultaneously.

1. Copy the template into `hpc/jobs/<model>/`
2. Fill in `<BASE_PATH>`, `<MODEL>`, `<MODEL_DIR>` at the top
3. Add one entry per quant to the `CONFIGS` array, replacing `<QUANT>` with the actual quant names
4. Update `--array=0-N` in the SBATCH header to match the number of configs (0-indexed)
5. Submit with:

```bash
sbatch hpc/jobs/<model>/your_build_array_job.sh
```

See [`hpc/jobs/Qwen25_7B/build_array_all.sh`](jobs/Qwen25_7B/build_array_all.sh) as a working example.

---

## Evaluation

Eval jobs invoke [`eval/custom_lmeval_wrapper.py`](../eval/custom_lmeval_wrapper.py) inside the container, which runs benchmarks via `lm-eval`. Refer to the [`eval/`](../eval) directory for task configs and documentation.

### Single eval job

For running one benchmark at a time, use [`hpc/jobs/eval_job_template.sh`](jobs/eval_job_template.sh) as a starting point:

1. Copy the template into `hpc/jobs/<model>/`
2. Fill in the placeholders at the top of the script:
   - `<BASE_PATH>` — path to the `trt-scripts` root
   - `<MODEL>` — model subfolder name used under `model/trt_engines/` (e.g. `Qwen25_3B`)
   - `<MODEL_DIR>` — HuggingFace model directory name (e.g. `Qwen2.5-3B`)
   - `<QUANT>` — quantization name (e.g. `W16A16`)
   - `<TASK>` — benchmark name matching a config file under `eval/configs/tasks/` (e.g. `mmlu`)
3. Submit with:

```bash
sbatch hpc/jobs/<model>/your_eval_job.sh
```

### Evaluating multiple benchmarks at once (array job)

To run all benchmarks for a given quant in parallel, use [`hpc/jobs/eval_array_job_template.sh`](jobs/eval_array_job_template.sh). Each array task picks a different benchmark config.

1. Copy the template into `hpc/jobs/<model>/`
2. Fill in `<BASE_PATH>`, `<MODEL>`, `<MODEL_DIR>`, `<QUANT>` at the top
3. The `CONFIGS` array lists all benchmark configs to run — add or remove entries as needed
4. Make sure the corresponding config files exist under `eval/configs/tasks/`
5. Submit with:

```bash
sbatch hpc/jobs/<model>/your_eval_array_job.sh
```

### Different SBATCH parameters per task (submit scripts)

SLURM array jobs share a single set of SBATCH parameters, so tasks with very different resource needs (e.g. time, memory, cores) cannot be handled in one `sbatch` call. In this case, use a submit script that fires multiple `sbatch` calls with different `--array`, `--time`, and resource overrides for each group of tasks.

See [`hpc/jobs/Qwen25_7B/W4A16_AWQ/submit_eval_W4A16_AWQ.sh`](jobs/Qwen25_7B/W4A16_AWQ/submit_eval_W4A16_AWQ.sh) for a working per-quant example, and [`hpc/jobs/Qwen25_7B/submit_eval_all.sh`](jobs/Qwen25_7B/submit_eval_all.sh) to run all quants in one go.

Run a submit script with:

```bash
./hpc/jobs/<model>/<quant>/submit_eval_<QUANT>.sh
```

### Monitor jobs

```bash
squeue --me
```

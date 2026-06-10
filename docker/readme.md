# docker/

Docker environment for running builds and evaluations locally. Based on `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc5`.

### Contents

- [Getting Started](#getting-started)
- [Makefile Commands](#makefile-commands)
- [Running Jobs](#running-jobs)
- [jobs/](#jobs)

---

## Getting Started

Build the image from the [Dockerfile](Dockerfile):

```bash
make build
```

Launch a detached container with host networking:

```bash
make run-shell-detached-hosted
```

Then attach to it:

```bash
docker exec -it trt_shell bash
```

The container mounts:
- `~/.cache/huggingface` → `/root/.cache/huggingface` — HuggingFace model cache
- `trtllm_data` (Docker volume) → `/workspace` — persistent workspace

---

## Makefile Commands

| Command | Description |
|---|---|
| `make build` | Build the Docker image |
| `make clean` | Remove dangling Docker images |
| `make run-shell` | Launch an interactive shell with GPU access |
| `make run-jupyter` | Launch JupyterLab on port `25566` |
| `make run-shell-detached` | Launch a detached shell container |
| `make run-shell-detached-hosted` | Launch a detached shell using host networking |

---

## Running Jobs

Jobs are run directly inside the container. From a shell, invoke the build or eval scripts directly:

```bash
bash /workspace/trt-scripts/docker/jobs/<model>/your_job.sh
```

---

## `jobs/`

Job scripts for running builds, evals, and benchmarks inside the container.

```
jobs/
  build_job_template.sh              # single-engine build template
  eval_job_template.sh               # single-task eval template
  eval_all_W16A16.sh                 # run all eval tasks for W16A16 (serial)
  <Model>/
    build_all_<model>_from_checkpoints.sh   # rebuild all quant bench engines from checkpoints
    bench_all_<model>.sh                    # benchmark all quant variants × 4 datasets
```

Build scripts load pre-built checkpoints from `model/trt_checkpoints/<Model>/<QUANT>/` and compile bench engines (no logits, batch 1). Benchmark scripts loop over all quant variants and all four synthetic datasets.

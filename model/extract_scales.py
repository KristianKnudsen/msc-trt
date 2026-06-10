import argparse
import json
import os
import re

import torch
from safetensors import safe_open


SCALE_KEYWORDS = ("scale", "scaling_factor", "kv_cache", "prequant")


def _is_scale_key(key: str) -> bool:
    return any(kw in key for kw in SCALE_KEYWORDS)


def _layer_index(key: str) -> str | None:
    m = re.search(r"\.layers\.(\d+)\.", key)
    return m.group(1) if m else None


def _suffix(key: str) -> str:
    parts = key.split(".")
    try:
        idx = parts.index("layers")
        return ".".join(parts[idx + 2:])
    except ValueError:
        return key


def _r(v: float, n: int = 6) -> float:
    return round(v, n)


def _tensor_stats(t: torch.Tensor) -> dict:
    flat = t.flatten().float()
    n = flat.numel()

    if n == 1:
        return {"value": _r(flat.item()), "n_scalars": 1}

    mn  = flat.min().item()
    mx  = flat.max().item()
    avg = flat.mean().item()
    med = flat.median().item()
    std = flat.std().item()
    cv  = (std / avg) if avg != 0 else float("inf")
    dr  = (mx / mn)   if mn != 0 else float("inf")
    # values more than 3 std above the mean — potential outlier channels
    n_outliers = int((flat > avg + 3 * std).sum().item())

    return {
        "min":           _r(mn),
        "max":           _r(mx),
        "mean":          _r(avg),
        "median":        _r(med),
        "std":           _r(std),
        "cv":            _r(cv),            # spread relative to magnitude
        "dynamic_range": _r(dr),            # max/min ratio — quantization difficulty
        "n_outliers":    n_outliers,         # channels > mean+3σ
        "n_scalars":     n,
    }


def extract_checkpoint(checkpoint_dir: str) -> dict:
    safetensors_files = sorted(
        f for f in os.listdir(checkpoint_dir) if f.endswith(".safetensors")
    )
    if not safetensors_files:
        return {}

    config_path = os.path.join(checkpoint_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # layer_idx (str) -> scale_suffix -> raw tensor
    raw: dict[str, dict[str, torch.Tensor]] = {}

    for fname in safetensors_files:
        fpath = os.path.join(checkpoint_dir, fname)
        with safe_open(fpath, framework="pt") as st:
            for key in st.keys():
                if not _is_scale_key(key):
                    continue
                layer_idx = _layer_index(key)
                if layer_idx is None:
                    continue
                suffix = _suffix(key)
                raw.setdefault(layer_idx, {})[suffix] = st.get_tensor(key).float()

    if not raw:
        return {}

    # Per-layer stats
    layers: dict[str, dict] = {}
    for layer_idx in sorted(raw.keys(), key=int):
        layers[layer_idx] = {
            suffix: _tensor_stats(tensor)
            for suffix, tensor in raw[layer_idx].items()
        }

    # Cross-layer summary per scale type
    all_suffixes = sorted({s for l in raw.values() for s in l})
    scales: dict[str, dict] = {}
    total_scalars = 0

    for suffix in all_suffixes:
        layer_indices = sorted(
            (i for i in raw if suffix in raw[i]), key=int
        )
        tensors = [raw[i][suffix] for i in layer_indices]
        total_scalars += sum(t.numel() for t in tensors)

        is_scalar = tensors[0].numel() == 1

        if is_scalar:
            vals = [_r(t.item()) for t in tensors]
            # skip constant tensors (e.g. uncalibrated scale_to_int = 1.0 everywhere)
            if min(vals) == max(vals):
                continue
            paired = sorted(zip(vals, layer_indices), reverse=True)
            scales[suffix] = {
                "min":  _r(min(vals)),
                "max":  _r(max(vals)),
                "mean": _r(sum(vals) / len(vals)),
                "top3": [[int(li), v] for v, li in paired[:3]],
            }
        else:
            layer_means = [t.mean().item() for t in tensors]
            layer_maxes = [t.max().item()  for t in tensors]
            layer_cvs   = [
                (t.std() / t.mean()).item() if t.mean().item() != 0 else float("inf")
                for t in tensors
            ]
            top3 = sorted(zip(layer_maxes, layer_indices), reverse=True)[:3]
            scales[suffix] = {
                "gmin": _r(min(t.min().item() for t in tensors)),
                "gmax": _r(max(t.max().item() for t in tensors)),
                "mean": _r(sum(layer_means) / len(layer_means)),
                "cv":   _r(sum(layer_cvs) / len(layer_cvs)),
                "top3": [[int(li), _r(v)] for v, li in top3],
            }

    config_slim = {
        "architecture": config.get("architecture"),
        "dtype":        config.get("dtype"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size":  config.get("hidden_size"),
        "quantization": config.get("quantization"),
    }

    summary = {
        "n_layers":      len(layers),
        "total_scalars": total_scalars,
        "scale_types":   all_suffixes,
        "scales":        scales,
    }

    return {"config": config_slim, "summary": summary}


def _compact_json(obj: dict) -> str:
    s = json.dumps(obj, indent=2)
    # Collapse [[int, float], ...] top3 arrays onto a single line
    s = re.sub(
        r'\[\s*(?:\[\s*\d+\s*,\s*[\d.e+\-]+\s*\]\s*,?\s*)+\]',
        lambda m: '[' + ', '.join(
            f'[{a}, {b}]' for a, b in re.findall(r'\[\s*(\d+)\s*,\s*([\d.e+\-]+)\s*\]', m.group(0))
        ) + ']',
        s
    )
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True,
                        help="Root dir containing <model>/<quant> subdirectories")
    parser.add_argument("--output", required=True,
                        help="Output JSON file path")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Limit to specific model names (default: all)")
    parser.add_argument("--quants", nargs="*", default=None,
                        help="Limit to specific quant names (default: all)")
    args = parser.parse_args()

    root = args.checkpoint_root
    results = {}
    skipped = []

    model_dirs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and (args.models is None or d in args.models)
    )

    for model in model_dirs:
        model_path = os.path.join(root, model)
        quant_dirs = sorted(
            d for d in os.listdir(model_path)
            if os.path.isdir(os.path.join(model_path, d))
            and (args.quants is None or d in args.quants)
        )
        for quant in quant_dirs:
            key = f"{model}/{quant}"
            ckpt_path = os.path.join(model_path, quant)
            print(f"Processing {key} ...", flush=True)
            data = extract_checkpoint(ckpt_path)
            if not data:
                print(f"  skipped (no safetensors)")
                skipped.append(key)
                continue
            s = data["summary"]
            print(f"  {s['n_layers']} layers, {s['total_scalars']:,} total scalars")
            print(f"  scale types: {s['scale_types']}")
            results[key] = data

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(_compact_json(results))

    print(f"\nWrote {len(results)} checkpoints to {args.output}")
    if skipped:
        print(f"Skipped (empty): {skipped}")


if __name__ == "__main__":
    main()

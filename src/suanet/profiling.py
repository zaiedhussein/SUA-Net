from __future__ import annotations

import copy
import csv
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, apply_overrides
from .model import build_model
from .utils import save_json


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    repetitions: int,
    passes: int = 1,
) -> dict[str, float]:
    device = inputs.device
    model.eval()
    if passes > 1:
        model.enable_mc_dropout()
    with torch.no_grad():
        for _ in range(warmup):
            for _ in range(passes):
                model(inputs)
        _synchronize(device)
        timings = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repetitions):
            for _ in range(iterations):
                _synchronize(device)
                start = time.perf_counter()
                for _ in range(passes):
                    model(inputs)
                _synchronize(device)
                timings.append((time.perf_counter() - start) * 1000.0)
        peak_memory = (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else float("nan")
        )
    values = np.asarray(timings)
    mean = float(values.mean())
    return {
        "mean_ms": mean,
        "sd_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "cases_per_second": float(1000.0 / mean),
        "peak_memory_mb": peak_memory,
        "timed_passes": int(len(values)),
    }


def _complexity(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[float, float]:
    try:
        from thop import profile
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "THOP is required for manuscript MAC/FLOP profiling; install project dependencies."
        ) from error
    model.eval()
    macs, _ = profile(model, inputs=(inputs,), verbose=False)
    return float(macs), float(2.0 * macs)


def profile_architectures(
    config: ExperimentConfig,
    variants: Sequence[Mapping],
    *,
    output_dir: str | Path,
    warmup: int = 50,
    iterations: int = 500,
    repetitions: int = 5,
    measure_runtime: bool = True,
    require_cuda: bool = True,
    mc_passes: Sequence[int] = (10, 20, 30, 50),
) -> dict:
    """Reproduce architecture complexity and controlled deployment profiling."""
    if warmup < 0 or iterations < 1 or repetitions < 1:
        raise ValueError("Invalid profiling iteration counts")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "The manuscript runtime benchmark requires CUDA. "
            "Use measure_runtime=false for hardware-independent complexity only."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    full_model = None
    inputs = torch.randn(1, 3, config.training.img_size, config.training.img_size, device=device)

    for variant in variants:
        variant_config = copy.deepcopy(config)
        apply_overrides(variant_config, {"model": dict(variant.get("model", {}))})
        model = build_model(variant_config.model, pretrained=False).to(device).float()
        counts = model.parameter_counts()
        macs, flops = _complexity(model, inputs)
        row = {
            "variant": str(variant["name"]),
            "trainable_parameters": counts["trainable"],
            "total_parameters": counts["total"],
            "parameters_millions": counts["total"] / 1e6,
            "macs": macs,
            "macs_giga": macs / 1e9,
            "flops": flops,
            "flops_giga": flops / 1e9,
        }
        if measure_runtime:
            row.update(
                _measure_forward(
                    model,
                    inputs,
                    warmup=warmup,
                    iterations=iterations,
                    repetitions=repetitions,
                )
            )
        rows.append(row)
        if all(
            bool(getattr(variant_config.model, key))
            for key in ("use_dla", "use_sva", "use_mgp")
        ):
            full_model = model
        else:
            del model

    mc_rows = []
    if measure_runtime:
        if full_model is None:
            full_model = build_model(config.model, pretrained=False).to(device).float()
        deterministic = _measure_forward(
            full_model,
            inputs,
            warmup=warmup,
            iterations=iterations,
            repetitions=repetitions,
            passes=1,
        )
        mc_rows.append({"strategy": "deterministic", "passes": 1, **deterministic})
        for passes in mc_passes:
            measured = _measure_forward(
                full_model,
                inputs,
                warmup=warmup,
                iterations=iterations,
                repetitions=repetitions,
                passes=int(passes),
            )
            measured["relative_cost"] = measured["mean_ms"] / deterministic["mean_ms"]
            mc_rows.append({"strategy": "mc_dropout", "passes": int(passes), **measured})

    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_mb": (
            torch.cuda.get_device_properties(device).total_memory / (1024**2)
            if device.type == "cuda"
            else None
        ),
        "precision": "FP32",
        "batch_size": 1,
        "input_resolution": config.training.img_size,
        "warmup": warmup,
        "iterations_per_repetition": iterations,
        "repetitions": repetitions,
        "latency_scope": "model forward only",
        "flop_convention": "FLOPs = 2 x MACs",
    }
    result = {"environment": environment, "architectures": rows, "mc_scaling": mc_rows}
    save_json(result, output / "profiling_results.json")
    for filename, records in (
        ("architecture_profile.csv", rows),
        ("mc_dropout_scaling.csv", mc_rows),
    ):
        if records:
            with (output / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return result

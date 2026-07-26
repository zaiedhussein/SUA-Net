from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from .config import apply_overrides, load_experiment_config
from .data import BreastUltrasoundDataset, build_transforms, make_folds, scan_dataset
from .engine import load_checkpoint_state, predict_loader
from .experiments import _make_loader, run_cross_validation
from .metrics import classification_metrics, summarize_metrics
from .model import build_model
from .utils import resolve_device, save_json, seed_everything


def run_configured_study(payload: Mapping) -> dict:
    """Run named configuration overrides with fixed folds for a single dataset."""
    base_config = load_experiment_config(payload["experiment_config"])
    output_dir = Path(payload.get("output_dir", "results/study"))
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = payload["variants"]
    results = {}
    for variant in variants:
        config = copy.deepcopy(base_config)
        apply_overrides(
            config,
            {
                section: values
                for section, values in variant.get("overrides", {}).items()
            },
        )
        slug = str(variant["name"]).lower().replace(" ", "_").replace("/", "_")
        config.experiment_name = f"{base_config.experiment_name}-{slug}"
        config.runtime.output_dir = str(output_dir / slug)
        result = run_cross_validation(config)
        results[str(variant["name"])] = {
            "overrides": variant.get("overrides", {}),
            "summary": result["summary"],
            "fold_metrics": result["fold_metrics"],
        }
    payload_out = {
        "dataset": base_config.dataset.name,
        "paired_folds": True,
        "variants": results,
    }
    save_json(payload_out, output_dir / "study_results.json")
    return payload_out


def run_multi_dataset_sensitivity(payload: Mapping) -> dict:
    """Run baseline, weighted-focal, and balanced-sampler sensitivity analyses."""
    output_dir = Path(payload.get("output_dir", "results/imbalance_sensitivity"))
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = payload.get(
        "strategies",
        ["baseline", "weighted_focal", "balanced_sampler"],
    )
    results = {}
    for dataset_name, config_path in payload["experiments"].items():
        base_config = load_experiment_config(config_path)
        results[dataset_name] = {}
        for strategy in strategies:
            config = copy.deepcopy(base_config)
            config.training.imbalance_strategy = str(strategy)
            config.training.use_mc_dropout = False
            config.runtime.output_dir = str(output_dir / dataset_name / str(strategy))
            config.validate()
            result = run_cross_validation(config)
            results[dataset_name][str(strategy)] = {
                "summary": result["summary"],
                "fold_metrics": result["fold_metrics"],
            }
    payload_out = {
        "paired_folds": True,
        "strategies": list(strategies),
        "datasets": results,
    }
    save_json(payload_out, output_dir / "imbalance_sensitivity.json")
    return payload_out


def run_mc_sample_sensitivity(payload: Mapping) -> dict:
    """Re-evaluate saved folds at T=10,20,30,50 without retraining."""
    sample_sizes = [int(value) for value in payload.get("sample_sizes", [10, 20, 30, 50])]
    output_dir = Path(payload.get("output_dir", "results/mc_sensitivity"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset_name, config_path in payload["configs"].items():
        config = load_experiment_config(config_path)
        result_dir = Path(payload["result_dirs"][dataset_name])
        device = resolve_device(config.runtime.device)
        seed_everything(config.training.seed, config.runtime.deterministic)
        samples = scan_dataset(config.dataset.root, config.dataset.name)
        folds = make_folds(
            samples,
            config.training.k_folds,
            config.training.seed,
            config.dataset.split_strategy,
        )
        fold_rows = {sample_size: [] for sample_size in sample_sizes}
        for fold_index, (_, validation_samples) in enumerate(folds, start=1):
            dataset = BreastUltrasoundDataset(
                validation_samples,
                config.training.img_size,
                transform=build_transforms(config.training.img_size, training=False),
                return_metadata=True,
            )
            loader = _make_loader(dataset, config.training.batch_size, False, config)
            model = build_model(config.model, pretrained=False).to(device)
            state_dict, _ = load_checkpoint_state(
                result_dir / f"fold_{fold_index}" / "best_model.pth",
                device,
            )
            model.load_state_dict(state_dict, strict=True)
            for sample_size in sample_sizes:
                output = predict_loader(
                    model,
                    loader,
                    device,
                    mixed_precision=config.training.mixed_precision,
                    mc_samples=sample_size,
                )
                metrics = classification_metrics(output.y_true, output.y_pred, output.y_prob)
                fold_rows[sample_size].append(
                    {
                        "fold": fold_index,
                        "auc_roc": metrics["auc_roc"],
                        "accuracy": metrics["accuracy"],
                        "mean_uncertainty": float(np.mean(output.uncertainty)),
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[dataset_name] = {
            f"T={sample_size}": {
                "fold_metrics": fold_rows[sample_size],
                "summary": summarize_metrics(
                    fold_rows[sample_size],
                    ["auc_roc", "accuracy", "mean_uncertainty"],
                ),
            }
            for sample_size in sample_sizes
        }
    payload_out = {
        "sample_sizes": sample_sizes,
        "classification_interpretation": (
            "AUC stability does not establish convergence of case-level uncertainty."
        ),
        "datasets": results,
    }
    save_json(payload_out, output_dir / "mc_sample_sensitivity.json")
    return payload_out

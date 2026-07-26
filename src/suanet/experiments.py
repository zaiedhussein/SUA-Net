from __future__ import annotations

import copy
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .analysis import paired_mc_across_datasets, threshold_operating_points
from .config import (
    ExperimentConfig,
    apply_overrides,
    load_experiment_config,
    save_resolved_config,
)
from .data import (
    BreastUltrasoundDataset,
    build_transforms,
    dataset_audit,
    make_folds,
    scan_dataset,
    validate_no_group_leakage,
)
from .engine import PredictionOutput, fit_fold, load_checkpoint_state, predict_loader
from .interpretability import (
    generate_gradcam_maps,
    save_ablation_gradcam_grid,
    save_failure_case_figure,
    save_uncertainty_case_figure,
    select_balanced_indices,
    select_failure_indices,
    select_high_uncertainty_indices,
)
from .losses import FocalLoss
from .metrics import bootstrap_auc_ci, classification_metrics, summarize_metrics
from .model import build_model
from .plots import (
    plot_ablation_delta,
    plot_ablation_effect_sizes,
    plot_ablation_heatmap,
    plot_ablation_results,
    plot_all_roc,
    plot_auc_matrix,
    plot_calibration,
    plot_confusion,
    plot_dataset_comparison,
    plot_fold_comparison,
    plot_fold_heatmap,
    plot_generalization_gaps,
    plot_metric_distribution,
    plot_metric_summary,
    plot_precision_recall,
    plot_radar,
    plot_roc,
    plot_significance_matrix,
    plot_training_history,
    plot_uncertainty_distributions,
)
from .reports import generalization_latex, write_ablation_report, write_training_report
from .statistics import ablation_statistics
from .utils import resolve_device, save_json, seed_everything

METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "auc_roc",
    "avg_prec",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
    "mcc",
    "kappa",
    "brier",
    "ece_10",
    "log_loss",
]

DEFAULT_ABLATIONS = [
    {"name": "Full Model", "use_dla": True, "use_sva": True, "use_mgp": True},
    {"name": "w/o SVA", "use_dla": True, "use_sva": False, "use_mgp": True},
    {"name": "w/o DLA", "use_dla": False, "use_sva": True, "use_mgp": True},
    {"name": "w/o MGP", "use_dla": True, "use_sva": True, "use_mgp": False},
    {"name": "w/o SVA+DLA", "use_dla": False, "use_sva": False, "use_mgp": True},
    {"name": "w/o SVA+MGP", "use_dla": True, "use_sva": False, "use_mgp": False},
    {"name": "w/o DLA+MGP", "use_dla": False, "use_sva": True, "use_mgp": False},
    {"name": "Backbone Only", "use_dla": False, "use_sva": False, "use_mgp": False},
]


def _variant_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_").replace("+", "_")


def _apply_model_variant(config: ExperimentConfig, variant: Mapping) -> None:
    """Apply architecture or design-ablation settings with key validation."""
    overrides = dict(variant.get("model", {}))
    for key in ("use_dla", "use_sva", "use_mgp"):
        if key in variant:
            overrides[key] = variant[key]
    if overrides:
        apply_overrides(config, {"model": overrides})


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    config: ExperimentConfig,
    *,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.runtime.num_workers > 0,
    )


def _save_predictions(output: PredictionOutput, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "image",
                "group_id",
                "y_true",
                "y_pred",
                "p_malignant",
                "mc_p_malignant",
                "uncertainty",
            ]
        )
        for index in range(len(output.y_true)):
            writer.writerow(
                [
                    output.sample_ids[index] if output.sample_ids else index,
                    output.image_paths[index] if output.image_paths else "",
                    output.group_ids[index] if output.group_ids else "",
                    int(output.y_true[index]),
                    int(output.y_pred[index]),
                    float(output.y_prob[index]),
                    float(output.mc_prob[index]) if output.mc_prob is not None else "",
                    float(output.uncertainty[index]) if output.uncertainty is not None else "",
                ]
            )


def _validate_audit(audit: Mapping, grouped: bool, require_verified_groups: bool = False) -> None:
    if audit["missing_files"]:
        raise FileNotFoundError(f"Dataset audit found {len(audit['missing_files'])} missing files")
    if audit["duplicate_paths"]:
        raise RuntimeError(
            f"Dataset audit found {len(audit['duplicate_paths'])} duplicate image paths"
        )
    if audit["duplicate_sample_ids"]:
        raise RuntimeError(
            f"Dataset audit found {len(audit['duplicate_sample_ids'])} duplicate sample IDs"
        )
    if grouped and audit["groups_with_conflicting_labels"]:
        raise RuntimeError(
            "Grouped evaluation requires one class per group; conflicting groups: "
            + ", ".join(audit["groups_with_conflicting_labels"][:5])
        )
    if require_verified_groups and audit.get("unverified_group_ids"):
        raise RuntimeError(
            "The manuscript protocol requires verified patient/case identifiers, but "
            f"{len(audit['unverified_group_ids'])} fallback group identifiers were found. "
            "Use the original metadata release containing the BUSBRA Case field."
        )


def _split_manifest(train_samples: Sequence[Mapping], val_samples: Sequence[Mapping]) -> dict:
    def records(samples: Sequence[Mapping]) -> list[dict]:
        return [
            {
                "sample_id": str(sample["sample_id"]),
                "group_id": str(sample.get("group_id", "")),
                "label": int(sample["label"]),
            }
            for sample in samples
        ]

    return {"train": records(train_samples), "validation": records(val_samples)}


def run_cross_validation(config: ExperimentConfig) -> dict:
    """Run the final-model protocol for one configured dataset."""
    config.validate()
    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    seed_everything(config.training.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)

    samples = scan_dataset(config.dataset.root, config.dataset.name)
    audit = dataset_audit(samples)
    save_json(audit, output_dir / "dataset_audit.json")
    grouped = config.dataset.split_strategy == "stratified_group"
    _validate_audit(audit, grouped, config.dataset.require_verified_groups)

    folds = make_folds(
        samples,
        config.training.k_folds,
        config.training.seed,
        config.dataset.split_strategy,
    )
    fold_metrics: list[dict[str, float]] = []
    fold_predictions: list[PredictionOutput] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    for fold_index, (train_samples, val_samples) in enumerate(folds, start=1):
        seed_everything(config.training.seed + fold_index, config.runtime.deterministic)
        if grouped:
            validate_no_group_leakage(train_samples, val_samples)
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(_split_manifest(train_samples, val_samples), fold_dir / "split_manifest.json")

        train_dataset = BreastUltrasoundDataset(
            train_samples,
            config.training.img_size,
            transform=build_transforms(
                config.training.img_size,
                training=True,
                strong=config.training.strong_augmentation,
            ),
        )
        val_dataset = BreastUltrasoundDataset(
            val_samples,
            config.training.img_size,
            transform=build_transforms(config.training.img_size, training=False),
            return_metadata=True,
        )
        sampler = None
        train_labels = np.asarray([int(sample["label"]) for sample in train_samples], dtype=int)
        class_counts = np.bincount(train_labels, minlength=2)
        if np.any(class_counts == 0):
            raise RuntimeError(f"Training fold is missing a class: {class_counts.tolist()}")
        if config.training.imbalance_strategy == "balanced_sampler":
            sample_weights = 1.0 / class_counts[train_labels]
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
        train_loader = _make_loader(
            train_dataset,
            config.training.batch_size,
            True,
            config,
            sampler=sampler,
        )
        val_loader = _make_loader(val_dataset, config.training.batch_size, False, config)

        model = build_model(config.model).to(device)
        class_weights = None
        if config.training.imbalance_strategy == "weighted_focal":
            class_weights = torch.as_tensor(
                len(train_labels) / (2.0 * class_counts),
                dtype=torch.float32,
                device=device,
            )
        criterion = FocalLoss(
            gamma=config.training.focal_gamma,
            label_smoothing=config.training.label_smoothing,
            class_weights=class_weights,
        ).to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, config.training.epochs // 3),
            T_mult=1,
            eta_min=1e-6,
        )
        result = fit_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=config.training.epochs,
            accumulation=config.training.accumulation,
            grad_clip=config.training.grad_clip,
            mixed_precision=config.training.mixed_precision,
            early_stopping=config.training.early_stopping,
            monitor=config.training.monitor,
            monitor_mode=config.training.monitor_mode,
            checkpoint_path=fold_dir / "best_model.pth",
            checkpoint_metadata={
                "experiment_name": config.experiment_name,
                "dataset": config.dataset.name,
                "fold": fold_index,
                "model": config.model.__dict__,
                "split_strategy": config.dataset.split_strategy,
            },
            use_tta=config.training.use_tta,
            use_mc_dropout=config.training.use_mc_dropout,
            mc_dropout_samples=config.training.mc_dropout_samples,
        )
        fold_metrics.append(result.metrics)
        fold_predictions.append(result.predictions)
        all_true.append(result.predictions.y_true)
        all_pred.append(result.predictions.y_pred)
        all_prob.append(result.predictions.y_prob)
        if config.runtime.save_predictions:
            _save_predictions(result.predictions, fold_dir / "predictions.csv")
        plot_training_history(result.history, fold_dir / "training_curves.png")
        plot_roc(
            result.predictions.y_true,
            result.predictions.y_prob,
            fold_dir / "roc_curve.png",
        )
        plot_precision_recall(
            result.predictions.y_true,
            result.predictions.y_prob,
            fold_dir / "pr_curve.png",
        )
        plot_confusion(
            result.predictions.y_true,
            result.predictions.y_pred,
            fold_dir / "confusion_matrix.png",
        )

    summary = summarize_metrics(fold_metrics, METRIC_KEYS)
    serializable = {
        "dataset": config.dataset.name,
        "protocol": {
            "split_strategy": config.dataset.split_strategy,
            "monitor": config.training.monitor,
            "monitor_mode": config.training.monitor_mode,
            "primary_inference": "tta" if config.training.use_tta else "deterministic",
            "mc_dropout_is_auxiliary": config.training.use_mc_dropout,
            "imbalance_strategy": config.training.imbalance_strategy,
        },
        "fold_metrics": fold_metrics,
        "summary": summary,
    }
    save_json(serializable, output_dir / "summary.json")

    pooled_true = np.concatenate(all_true)
    pooled_pred = np.concatenate(all_pred)
    pooled_prob = np.concatenate(all_prob)
    pooled_auc_ci = bootstrap_auc_ci(pooled_true, pooled_prob)
    serializable["pooled_auc_bootstrap_95_ci"] = {
        "ci_lo": pooled_auc_ci[0],
        "ci_hi": pooled_auc_ci[1],
        "resamples": 2000,
        "seed": 42,
    }
    threshold_rows = threshold_operating_points(pooled_true, pooled_prob)
    save_json(
        {
            "dataset": config.dataset.name,
            "probability_source": "pooled out-of-fold deterministic predictions",
            "threshold_grid_step": 0.05,
            "operating_points": threshold_rows,
        },
        output_dir / "threshold_analysis.json",
    )
    save_json(serializable, output_dir / "summary.json")
    plot_metric_summary(summary, output_dir / "metrics_summary.png")
    plot_fold_comparison(fold_metrics, output_dir / "fold_comparison.png")
    plot_metric_distribution(fold_metrics, output_dir / "metrics_boxplot.png", kind="box")
    plot_metric_distribution(fold_metrics, output_dir / "metrics_violin.png", kind="violin")
    plot_fold_heatmap(fold_metrics, output_dir / "fold_heatmap.png")
    plot_radar(summary, output_dir / "radar_chart.png")
    plot_roc(pooled_true, pooled_prob, output_dir / "aggregated_roc.png")
    plot_precision_recall(pooled_true, pooled_prob, output_dir / "aggregated_pr.png")
    plot_confusion(
        pooled_true,
        pooled_pred,
        output_dir / "aggregated_confusion_matrix.png",
    )
    plot_calibration(pooled_true, pooled_prob, output_dir / "calibration_curve.png")
    write_training_report(
        dataset_name=config.dataset.name,
        fold_metrics=fold_metrics,
        summary=summary,
        path=output_dir / "academic_report.txt",
    )
    return {
        "fold_metrics": fold_metrics,
        "fold_predictions": fold_predictions,
        "summary": summary,
        "audit": audit,
    }


def run_training_suite(payload: Mapping) -> dict:
    """Run the three configured final-model experiments and a global comparison."""
    experiments = payload["experiments"]
    output_dir = Path(payload.get("output_dir", "results/all_datasets"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    summaries = {}
    prediction_outputs = {}
    for dataset_name, config_path in experiments.items():
        config = load_experiment_config(config_path)
        result = run_cross_validation(config)
        results[dataset_name] = {
            "fold_metrics": result["fold_metrics"],
            "summary": result["summary"],
            "audit": result["audit"],
        }
        summaries[dataset_name] = result["summary"]
        prediction_outputs[dataset_name] = result["fold_predictions"]
    save_json({"datasets": results}, output_dir / "all_datasets_summary.json")
    if all(
        all(output.mc_prob is not None for output in outputs)
        for outputs in prediction_outputs.values()
    ):
        mc_analysis = paired_mc_across_datasets(
            prediction_outputs,
            n_bootstrap=2000,
            seed=42,
        )
        save_json(mc_analysis, output_dir / "mc_dropout_paired_analysis.json")
    plot_dataset_comparison(summaries, output_dir / "cross_dataset_comparison.png")
    lines = ["SUA-Net multi-dataset comparison", ""]
    for dataset_name, summary in summaries.items():
        lines.append(
            f"{dataset_name}: AUC={summary['auc_roc']['mean']:.4f} ± "
            f"{summary['auc_roc']['std']:.4f}; accuracy={summary['accuracy']['mean']:.4f}"
        )
    (output_dir / "academic_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"datasets": results}


def _ablation_gradcam(
    *,
    config: ExperimentConfig,
    variants: Sequence[Mapping],
    root_dir: Path,
    max_cases: int,
) -> Path:
    samples = scan_dataset(config.dataset.root, config.dataset.name)
    folds = make_folds(
        samples,
        config.training.k_folds,
        config.training.seed,
        config.dataset.split_strategy,
    )
    val_samples = folds[0][1]
    dataset = BreastUltrasoundDataset(
        val_samples,
        config.training.img_size,
        transform=build_transforms(config.training.img_size, training=False),
        return_metadata=True,
    )
    indices = select_balanced_indices(val_samples, max_cases)
    if not indices:
        raise RuntimeError("No validation cases were available for ablation Grad-CAM")
    device = resolve_device(config.runtime.device)
    heatmaps = {}
    for variant in variants:
        variant_config = copy.deepcopy(config)
        variant_name = str(variant["name"])
        _apply_model_variant(variant_config, variant)
        model = build_model(variant_config.model, pretrained=False).to(device)
        checkpoint = root_dir / _variant_slug(variant_name) / "fold_1" / "best_model.pth"
        state_dict, _ = load_checkpoint_state(checkpoint, device)
        model.load_state_dict(state_dict)
        heatmaps[variant_name] = generate_gradcam_maps(
            model=model,
            dataset=dataset,
            indices=indices,
            device=device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = root_dir / "gradcam_ablation_comparison.png"
    save_ablation_gradcam_grid(
        dataset=dataset,
        indices=indices,
        heatmaps=heatmaps,
        path=output,
    )
    return output


def run_ablation(
    config: ExperimentConfig,
    variants: Sequence[Mapping] | None = None,
    *,
    generate_gradcam: bool = True,
    gradcam_samples: int = 8,
) -> dict:
    """Run the fixed-fold ablation protocol with complete paired statistics."""
    config.validate()
    root_dir = Path(config.runtime.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    selected_variants = list(variants or DEFAULT_ABLATIONS)
    all_results = {}
    summaries = {}
    parameter_counts = {}

    for variant in selected_variants:
        variant_config = copy.deepcopy(config)
        variant_name = str(variant["name"])
        variant_config.experiment_name = f"{config.experiment_name}-{_variant_slug(variant_name)}"
        variant_config.runtime.output_dir = str(root_dir / _variant_slug(variant_name))
        _apply_model_variant(variant_config, variant)
        probe_model = build_model(variant_config.model, pretrained=False)
        parameter_counts[variant_name] = probe_model.parameter_counts()
        del probe_model
        result = run_cross_validation(variant_config)
        all_results[variant_name] = result
        summaries[variant_name] = result["summary"]

    fold_metrics_by_variant = {
        variant_name: result["fold_metrics"] for variant_name, result in all_results.items()
    }
    predictions_by_variant = {
        variant_name: result["fold_predictions"] for variant_name, result in all_results.items()
    }
    reference_name = str(selected_variants[0]["name"])
    statistics = ablation_statistics(
        fold_metrics_by_variant,
        prediction_outputs=predictions_by_variant,
        reference_name=reference_name,
    )
    serializable_results = {
        variant_name: {
            "fold_metrics": result["fold_metrics"],
            "summary": result["summary"],
            "audit": result["audit"],
        }
        for variant_name, result in all_results.items()
    }
    payload = {
        "protocol": {
            "monitor": config.training.monitor,
            "monitor_mode": config.training.monitor_mode,
            "paired_folds": True,
            "primary_inference": "deterministic",
            "reference_variant": reference_name,
        },
        "variants": selected_variants,
        "parameter_counts": parameter_counts,
        "results": serializable_results,
        "statistical_tests": statistics,
    }
    save_json(payload, root_dir / "ablation_results.json")
    save_json(statistics, root_dir / "ablation_statistics.json")
    plot_ablation_results(summaries, root_dir / "ablation_comparison.png")
    plot_ablation_delta(summaries, root_dir / "ablation_delta.png")
    plot_ablation_heatmap(summaries, root_dir / "ablation_heatmap.png")
    plot_ablation_effect_sizes(
        statistics,
        root_dir / "ablation_effect_sizes_auc.png",
    )
    plot_significance_matrix(
        statistics,
        root_dir / "significance_matrix_auc.png",
        metric="auc_roc",
    )
    plot_significance_matrix(
        statistics,
        root_dir / "significance_matrix_f1.png",
        metric="f1",
    )
    write_ablation_report(
        summaries=summaries,
        parameter_counts=parameter_counts,
        statistics=statistics,
        path=root_dir / "ablation_report.txt",
    )
    if generate_gradcam:
        payload["gradcam_figure"] = str(
            _ablation_gradcam(
                config=config,
                variants=selected_variants,
                root_dir=root_dir,
                max_cases=gradcam_samples,
            )
        )
        save_json(payload, root_dir / "ablation_results.json")
    return payload


def _fold_index(path: Path) -> int:
    try:
        return int(path.parent.name.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid fold checkpoint path: {path}") from error


def _load_models_from_directory(
    config: ExperimentConfig,
    model_dir: str | Path,
    device: torch.device,
    *,
    expected_folds: int | None,
) -> list[torch.nn.Module]:
    model_dir = Path(model_dir)
    checkpoint_paths = sorted(
        model_dir.glob("fold_*/best_model.pth"),
        key=_fold_index,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold checkpoints found under {model_dir}")
    if expected_folds is not None and len(checkpoint_paths) != expected_folds:
        raise RuntimeError(
            f"Expected {expected_folds} fold checkpoints under {model_dir}, "
            f"found {len(checkpoint_paths)}"
        )

    models = []
    for checkpoint_path in checkpoint_paths:
        model = build_model(config.model, pretrained=False).to(device)
        state_dict, checkpoint = load_checkpoint_state(checkpoint_path, device)
        checkpoint_model = checkpoint.get("metadata", {}).get("model")
        if checkpoint_model and checkpoint_model != config.model.__dict__:
            raise RuntimeError(
                f"Model configuration does not match checkpoint metadata: {checkpoint_path}"
            )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        models.append(model)
    return models


def ensemble_predict(
    models: Sequence[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    *,
    mc_samples: int = 0,
) -> PredictionOutput:
    """Average fold probabilities after optional per-fold MC-dropout inference."""
    if not models:
        raise ValueError("At least one model is required for ensemble inference")
    outputs = [
        predict_loader(
            model,
            loader,
            device,
            mixed_precision=mixed_precision,
            use_tta=False,
            mc_samples=mc_samples,
        )
        for model in models
    ]
    for output in outputs[1:]:
        if not np.array_equal(outputs[0].y_true, output.y_true):
            raise RuntimeError("Fold models returned predictions in different sample order")
        if outputs[0].sample_ids != output.sample_ids:
            raise RuntimeError("Fold models returned different sample identifiers")
        if outputs[0].group_ids != output.group_ids:
            raise RuntimeError("Fold models returned different group identifiers")
    probabilities = np.stack([output.y_prob for output in outputs])
    mean_probability = probabilities.mean(axis=0)
    prediction = (mean_probability >= 0.5).astype(int)
    ddof = 1 if len(models) > 1 else 0
    return PredictionOutput(
        y_true=outputs[0].y_true,
        y_pred=prediction,
        y_prob=mean_probability,
        uncertainty=probabilities.std(axis=0, ddof=ddof),
        sample_ids=outputs[0].sample_ids,
        image_paths=outputs[0].image_paths,
        group_ids=outputs[0].group_ids,
    )


def _load_within_metrics(
    model_dirs: Mapping[str, str | Path],
    explicit_auc: Mapping | None,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for dataset_name, model_dir in model_dirs.items():
        summary_path = Path(model_dir) / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        values[dataset_name] = {
            metric: float(summary["mean"])
            for metric, summary in payload.get("summary", {}).items()
            if isinstance(summary, Mapping) and summary.get("mean") is not None
        }
    for dataset_name, auc in (explicit_auc or {}).items():
        if float(auc) > 0:
            values.setdefault(str(dataset_name), {})["auc_roc"] = float(auc)
    return values


def _load_within_auc(
    model_dirs: Mapping[str, str | Path],
    explicit: Mapping | None,
) -> dict[str, float]:
    """Backward-compatible AUC view over the complete within-metric loader."""
    metrics = _load_within_metrics(model_dirs, explicit)
    return {
        dataset_name: values["auc_roc"]
        for dataset_name, values in metrics.items()
        if "auc_roc" in values
    }


def run_generalization(payload: Mapping) -> dict:
    """Run all configured source-to-target fold-ensemble evaluations."""
    datasets = payload["datasets"]
    model_dirs = payload["model_dirs"]
    base_config = load_experiment_config(payload["base_config"])
    source_config_paths = payload.get("source_configs", {})
    source_configs = {
        source: load_experiment_config(source_config_paths[source])
        if source in source_config_paths
        else copy.deepcopy(base_config)
        for source in model_dirs
    }
    output_dir = Path(payload.get("output_dir", "results/generalization"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(base_config.training.seed, base_config.runtime.deterministic)
    device = resolve_device(base_config.runtime.device)
    strict_folds = bool(payload.get("strict_fold_count", True))
    mc_samples_override = payload.get("mc_dropout_samples")

    loaded_models = {
        source: _load_models_from_directory(
            source_configs[source],
            model_dir,
            device,
            expected_folds=source_configs[source].training.k_folds if strict_folds else None,
        )
        for source, model_dir in model_dirs.items()
    }
    within_metrics = _load_within_metrics(model_dirs, payload.get("within_auc"))
    within_auc = {
        dataset_name: metrics["auc_roc"]
        for dataset_name, metrics in within_metrics.items()
        if "auc_roc" in metrics
    }
    cross_metrics = {}
    prediction_outputs = {}

    for source, models in loaded_models.items():
        source_config = source_configs[source]
        if mc_samples_override is None:
            mc_samples = (
                source_config.training.mc_dropout_samples
                if source_config.training.use_mc_dropout
                else 0
            )
        else:
            mc_samples = int(mc_samples_override)
        for target, root in datasets.items():
            if source == target and not payload.get("include_diagonal", False):
                continue
            samples = scan_dataset(root, target)
            audit = dataset_audit(samples)
            _validate_audit(audit, grouped=False)
            dataset = BreastUltrasoundDataset(
                samples,
                source_config.training.img_size,
                transform=build_transforms(
                    source_config.training.img_size,
                    training=False,
                ),
                return_metadata=True,
            )
            loader = _make_loader(
                dataset,
                source_config.training.batch_size,
                False,
                source_config,
            )
            output = ensemble_predict(
                models,
                loader,
                device,
                source_config.training.mixed_precision,
                mc_samples=mc_samples,
            )
            metrics = classification_metrics(
                output.y_true,
                output.y_pred,
                output.y_prob,
            )
            ci_low, ci_high = bootstrap_auc_ci(output.y_true, output.y_prob)
            metrics["auc_ci_lo"] = ci_low
            metrics["auc_ci_hi"] = ci_high
            metrics["mean_uncertainty"] = float(output.uncertainty.mean())
            metrics["ensemble_folds"] = len(models)
            metrics["mc_dropout_samples"] = mc_samples
            source_baseline = within_metrics.get(source, {})
            for metric_name, suffix in (
                ("auc_roc", "auc"),
                ("f1", "f1"),
                ("mcc", "mcc"),
                ("sensitivity", "sensitivity"),
            ):
                if metric_name in source_baseline:
                    metrics[f"delta_{suffix}"] = (
                        float(source_baseline[metric_name]) - float(metrics[metric_name])
                    )
            pair = f"{source}->{target}"
            pair_dir = output_dir / f"{source}_to_{target}".replace("-", "_")
            _save_predictions(output, pair_dir / "predictions.csv")
            save_json(
                {"dataset_audit": audit, "metrics": metrics},
                pair_dir / "metrics.json",
            )
            plot_roc(
                output.y_true,
                output.y_prob,
                pair_dir / "roc_curve.png",
                title=pair,
            )
            plot_confusion(
                output.y_true,
                output.y_pred,
                pair_dir / "confusion_matrix.png",
                title=pair,
            )
            cross_metrics[pair] = metrics
            prediction_outputs[pair] = output

    if not cross_metrics:
        raise RuntimeError("No source-to-target generalization pairs were evaluated")
    aucs = np.asarray([metrics["auc_roc"] for metrics in cross_metrics.values()])
    accuracies = np.asarray([metrics["accuracy"] for metrics in cross_metrics.values()])
    valid_within = {dataset_name: score for dataset_name, score in within_auc.items() if score > 0}
    valid_gaps = [
        valid_within[pair.split("->")[0]] - metrics["auc_roc"]
        for pair, metrics in cross_metrics.items()
        if pair.split("->")[0] in valid_within
    ]
    summary = {
        "mean_cross_auc": float(aucs.mean()),
        "std_cross_auc": float(aucs.std(ddof=0)),
        "mean_cross_accuracy": float(accuracies.mean()),
        "std_cross_accuracy": float(accuracies.std(ddof=0)),
        "mean_generalization_gap": float(np.mean(valid_gaps)) if valid_gaps else None,
    }
    result = {
        "protocol": {
            "fold_ensemble": True,
            "strict_fold_count": strict_folds,
            "mc_dropout": any(
                int(metrics.get("mc_dropout_samples", 0)) > 0
                for metrics in cross_metrics.values()
            ),
        },
        "within_auc": within_auc,
        "within_metrics": within_metrics,
        "cross_dataset": cross_metrics,
        "summary": summary,
    }
    save_json(result, output_dir / "generalization_results.json")

    matrix = dict(cross_metrics)
    for dataset_name, score in within_auc.items():
        matrix[f"{dataset_name}->{dataset_name}"] = {"auc_roc": score}
    plot_auc_matrix(matrix, output_dir / "auc_matrix.png")
    if valid_within:
        valid_cross = {
            pair: metrics
            for pair, metrics in cross_metrics.items()
            if pair.split("->")[0] in valid_within
        }
        if valid_cross:
            plot_generalization_gaps(
                valid_within,
                valid_cross,
                output_dir / "generalization_gap.png",
            )
    plot_all_roc(
        prediction_outputs,
        output_dir / "all_roc_curves.png",
        title="Cross-dataset ROC curves — all pairs",
    )
    plot_uncertainty_distributions(
        prediction_outputs,
        output_dir / "uncertainty_distribution.png",
    )
    latex = generalization_latex(
        dataset_names=list(datasets),
        cross_metrics=cross_metrics,
        within_auc=within_auc,
    )
    (output_dir / "latex_tables.tex").write_text(latex + "\n", encoding="utf-8")
    return result


def generate_failure_cases(
    config: ExperimentConfig,
    fold_index: int,
    max_cases: int = 6,
) -> dict:
    output_dir = Path(config.runtime.output_dir)
    device = resolve_device(config.runtime.device)
    seed_everything(config.training.seed, config.runtime.deterministic)
    samples = scan_dataset(config.dataset.root, config.dataset.name)
    folds = make_folds(
        samples,
        config.training.k_folds,
        config.training.seed,
        config.dataset.split_strategy,
    )
    if not 1 <= fold_index <= len(folds):
        raise ValueError(f"fold_index must be between 1 and {len(folds)}")
    _, val_samples = folds[fold_index - 1]
    dataset = BreastUltrasoundDataset(
        val_samples,
        config.training.img_size,
        transform=build_transforms(config.training.img_size, training=False),
        return_metadata=True,
    )
    loader = _make_loader(dataset, config.training.batch_size, False, config)
    model = build_model(config.model, pretrained=False).to(device)
    checkpoint_path = output_dir / f"fold_{fold_index}" / "best_model.pth"
    state_dict, checkpoint = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    predictions = predict_loader(
        model,
        loader,
        device,
        mixed_precision=config.training.mixed_precision,
        use_tta=config.training.use_tta,
        mc_samples=0,
    )
    indices = select_failure_indices(
        predictions.y_true,
        predictions.y_pred,
        predictions.y_prob,
        max_cases=max_cases,
    )
    figure_path = output_dir / f"fold_{fold_index}" / "failure_cases_gradcam.png"
    save_failure_case_figure(
        model=model,
        dataset=dataset,
        prediction_output=predictions,
        indices=indices,
        device=device,
        path=figure_path,
        dataset_name=config.dataset.name,
    )
    cases = [
        {
            "index": index,
            "sample_id": predictions.sample_ids[index] if predictions.sample_ids else index,
            "image": predictions.image_paths[index] if predictions.image_paths else "",
            "y_true": int(predictions.y_true[index]),
            "y_pred": int(predictions.y_pred[index]),
            "p_malignant": float(predictions.y_prob[index]),
        }
        for index in indices
    ]
    save_json(cases, figure_path.with_suffix(".json"))
    return {"figure": str(figure_path), "cases": cases, "checkpoint": checkpoint}


def generate_uncertainty_cases(
    config: ExperimentConfig,
    fold_index: int,
    max_cases: int = 6,
) -> dict:
    """Generate held-out high-uncertainty cases with Grad-CAM and SVA variance."""
    output_dir = Path(config.runtime.output_dir)
    device = resolve_device(config.runtime.device)
    seed_everything(config.training.seed, config.runtime.deterministic)
    samples = scan_dataset(config.dataset.root, config.dataset.name)
    folds = make_folds(
        samples,
        config.training.k_folds,
        config.training.seed,
        config.dataset.split_strategy,
    )
    if not 1 <= fold_index <= len(folds):
        raise ValueError(f"fold_index must be between 1 and {len(folds)}")
    _, validation_samples = folds[fold_index - 1]
    dataset = BreastUltrasoundDataset(
        validation_samples,
        config.training.img_size,
        transform=build_transforms(config.training.img_size, training=False),
        return_metadata=True,
    )
    loader = _make_loader(dataset, config.training.batch_size, False, config)
    model = build_model(config.model, pretrained=False).to(device)
    checkpoint_path = output_dir / f"fold_{fold_index}" / "best_model.pth"
    state_dict, checkpoint = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    predictions = predict_loader(
        model,
        loader,
        device,
        mixed_precision=config.training.mixed_precision,
        use_tta=False,
        mc_samples=config.training.mc_dropout_samples,
    )
    indices = select_high_uncertainty_indices(predictions.uncertainty, max_cases=max_cases)
    figure_path = (
        output_dir / f"fold_{fold_index}" / "high_uncertainty_examples.png"
    )
    save_uncertainty_case_figure(
        model=model,
        dataset=dataset,
        prediction_output=predictions,
        indices=indices,
        device=device,
        path=figure_path,
        dataset_name=config.dataset.name,
    )
    cases = [
        {
            "index": index,
            "sample_id": predictions.sample_ids[index] if predictions.sample_ids else index,
            "image": predictions.image_paths[index] if predictions.image_paths else "",
            "group_id": predictions.group_ids[index] if predictions.group_ids else "",
            "y_true": int(predictions.y_true[index]),
            "y_pred": int(predictions.y_pred[index]),
            "mc_p_malignant": float(predictions.y_prob[index]),
            "uncertainty": float(predictions.uncertainty[index]),
        }
        for index in indices
    ]
    save_json(cases, figure_path.with_suffix(".json"))
    return {"figure": str(figure_path), "cases": cases, "checkpoint": checkpoint}
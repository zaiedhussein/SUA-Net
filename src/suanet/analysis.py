from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from .engine import PredictionOutput
from .metrics import classification_metrics

INFERENTIAL_METRICS = ("auc_roc", "f1", "mcc", "brier", "ece_10")


def holm_correction(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down correction with monotonic adjusted p-values."""
    if not p_values:
        return {}
    ordered = sorted(p_values, key=lambda key: float(p_values[key]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min((count - rank) * float(p_values[key]), 1.0)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _metrics_from_probability(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return classification_metrics(y_true, prediction, probability)


def threshold_operating_points(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    step: float = 0.05,
    sensitivity_target: float = 0.90,
) -> list[dict[str, float | int | str | list[str]]]:
    """Evaluate manuscript operating-point rules on a fixed threshold grid."""
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")
    if not 0 < sensitivity_target <= 1:
        raise ValueError("sensitivity_target must be in (0, 1]")
    true = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    thresholds = np.round(np.arange(0.0, 1.0 + step / 2.0, step), 10)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = classification_metrics(
            true,
            (probability >= threshold).astype(int),
            probability,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "mcc": metrics["mcc"],
                "youden_j": metrics["youden_j"],
                "tp": int(metrics["tp"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
            }
        )

    def best(metric: str) -> dict[str, float | int]:
        return max(
            rows,
            key=lambda row: (
                float(row[metric]),
                float(row["accuracy"]),
                -abs(float(row["threshold"]) - 0.5),
            ),
        )

    default = min(rows, key=lambda row: abs(float(row["threshold"]) - 0.5))
    target_candidates = [
        row for row in rows if float(row["sensitivity"]) + 1e-12 >= sensitivity_target
    ]
    target = (
        max(
            target_candidates,
            key=lambda row: (
                float(row["specificity"]),
                float(row["accuracy"]),
                float(row["threshold"]),
            ),
        )
        if target_candidates
        else max(rows, key=lambda row: float(row["sensitivity"]))
    )
    selections = [
        ("Default", default),
        ("Max MCC", best("mcc")),
        ("Max Youden", best("youden_j")),
        (f"Sensitivity >= {sensitivity_target:.2f}", target),
    ]
    grouped: dict[float, dict] = {}
    for rule, row in selections:
        threshold = float(row["threshold"])
        if threshold not in grouped:
            grouped[threshold] = {**row, "rules": []}
        grouped[threshold]["rules"].append(rule)
    result = []
    for _threshold, row in sorted(grouped.items(), key=lambda item: item[0], reverse=True):
        row["rule"] = " / ".join(row["rules"])
        result.append(row)
    return result


def exact_mcnemar(
    y_true: Sequence[int],
    deterministic_prediction: Sequence[int],
    mc_prediction: Sequence[int],
) -> dict[str, float | int | str]:
    true = np.asarray(y_true, dtype=int)
    deterministic = np.asarray(deterministic_prediction, dtype=int)
    stochastic = np.asarray(mc_prediction, dtype=int)
    if not (len(true) == len(deterministic) == len(stochastic)):
        raise ValueError("McNemar inputs must have equal lengths")
    det_wrong_mc_right = int(np.sum((deterministic != true) & (stochastic == true)))
    det_right_mc_wrong = int(np.sum((deterministic == true) & (stochastic != true)))
    discordant = det_wrong_mc_right + det_right_mc_wrong
    p_value = (
        float(
            binomtest(
                min(det_wrong_mc_right, det_right_mc_wrong),
                discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "deterministic_wrong_mc_right": det_wrong_mc_right,
        "deterministic_right_mc_wrong": det_right_mc_wrong,
        "discordant": discordant,
        "p_nominal": p_value,
        "method": "exact two-sided binomial McNemar test",
    }


def _paired_cluster_bootstrap(
    y_true: np.ndarray,
    deterministic_prob: np.ndarray,
    mc_prob: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float | int | list[float]]]:
    if not (
        len(y_true)
        == len(deterministic_prob)
        == len(mc_prob)
        == len(group_ids)
    ):
        raise ValueError("Paired bootstrap arrays must have equal lengths")
    units = np.unique(group_ids)
    unit_indices = {unit: np.flatnonzero(group_ids == unit) for unit in units}
    observed_det = _metrics_from_probability(y_true, deterministic_prob)
    observed_mc = _metrics_from_probability(y_true, mc_prob)
    differences: dict[str, list[float]] = defaultdict(list)
    generator = np.random.RandomState(seed)
    for _ in range(n_bootstrap):
        sampled_units = generator.choice(units, size=len(units), replace=True)
        indices = np.concatenate([unit_indices[unit] for unit in sampled_units])
        sampled_true = y_true[indices]
        if len(np.unique(sampled_true)) < 2:
            continue
        det_metrics = _metrics_from_probability(sampled_true, deterministic_prob[indices])
        mc_metrics = _metrics_from_probability(sampled_true, mc_prob[indices])
        for metric in INFERENTIAL_METRICS:
            difference = float(mc_metrics[metric] - det_metrics[metric])
            if np.isfinite(difference):
                differences[metric].append(difference)

    results = {}
    for metric in INFERENTIAL_METRICS:
        values = np.asarray(differences[metric], dtype=float)
        observed = float(observed_mc[metric] - observed_det[metric])
        if values.size:
            ci_lo, ci_hi = np.quantile(values, [0.025, 0.975])
            p_value = min(
                2.0 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0))),
                1.0,
            )
        else:
            ci_lo, ci_hi, p_value = float("nan"), float("nan"), 1.0
        results[metric] = {
            "deterministic": float(observed_det[metric]),
            "mc_20": float(observed_mc[metric]),
            "delta": observed,
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "p_nominal": float(p_value),
            "successful_resamples": int(values.size),
        }
    return results


def paired_mc_dataset(
    outputs: Sequence[PredictionOutput],
    *,
    dataset_name: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    if not outputs:
        raise ValueError("At least one fold prediction output is required")
    if any(output.mc_prob is None for output in outputs):
        raise ValueError("Every fold must contain MC-Dropout probabilities")
    y_true = np.concatenate([output.y_true for output in outputs])
    deterministic_prob = np.concatenate([output.y_prob for output in outputs])
    mc_prob = np.concatenate([np.asarray(output.mc_prob) for output in outputs])
    group_ids = []
    for fold_index, output in enumerate(outputs, start=1):
        if output.group_ids:
            group_ids.extend(output.group_ids)
        elif output.sample_ids:
            group_ids.extend(output.sample_ids)
        else:
            group_ids.extend(
                [f"fold_{fold_index}_sample_{index}" for index in range(len(output.y_true))]
            )
    groups = np.asarray(group_ids, dtype=object)
    metrics = _paired_cluster_bootstrap(
        y_true,
        deterministic_prob,
        mc_prob,
        groups,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    mcnemar = exact_mcnemar(
        y_true,
        (deterministic_prob >= 0.5).astype(int),
        (mc_prob >= 0.5).astype(int),
    )
    return {
        "dataset": dataset_name,
        "n_samples": int(len(y_true)),
        "n_resampling_units": int(len(np.unique(groups))),
        "resampling": "patient/group clustered" if len(np.unique(groups)) < len(groups) else "image",
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "threshold": 0.5,
        "metrics": metrics,
        "mcnemar": mcnemar,
    }


def paired_mc_across_datasets(
    outputs_by_dataset: Mapping[str, Sequence[PredictionOutput]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    datasets = {
        dataset_name: paired_mc_dataset(
            outputs,
            dataset_name=dataset_name,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for dataset_name, outputs in outputs_by_dataset.items()
    }
    for metric in INFERENTIAL_METRICS:
        adjusted = holm_correction(
            {
                dataset_name: float(result["metrics"][metric]["p_nominal"])
                for dataset_name, result in datasets.items()
            }
        )
        for dataset_name, p_value in adjusted.items():
            datasets[dataset_name]["metrics"][metric]["p_holm"] = p_value
    mcnemar_adjusted = holm_correction(
        {
            dataset_name: float(result["mcnemar"]["p_nominal"])
            for dataset_name, result in datasets.items()
        }
    )
    for dataset_name, p_value in mcnemar_adjusted.items():
        datasets[dataset_name]["mcnemar"]["p_holm"] = p_value
    return {
        "primary_endpoint": "auc_roc",
        "secondary_endpoints": ["f1", "mcc", "brier", "ece_10", "mcnemar"],
        "correction": "Holm within each three-dataset metric family",
        "datasets": datasets,
    }


def load_prediction_directory(path: str | Path) -> list[PredictionOutput]:
    """Load fold prediction CSV files produced by the public training command."""
    root = Path(path)
    outputs = []
    for csv_path in sorted(root.glob("fold_*/predictions.csv")):
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig")))
        if not rows:
            continue
        mc_values = [row.get("mc_p_malignant", "") for row in rows]
        outputs.append(
            PredictionOutput(
                y_true=np.asarray([int(row["y_true"]) for row in rows]),
                y_pred=np.asarray([int(row["y_pred"]) for row in rows]),
                y_prob=np.asarray([float(row["p_malignant"]) for row in rows]),
                mc_prob=(
                    np.asarray([float(value) for value in mc_values])
                    if all(value != "" for value in mc_values)
                    else None
                ),
                uncertainty=np.asarray(
                    [
                        float(row["uncertainty"])
                        if row.get("uncertainty", "") != ""
                        else np.nan
                        for row in rows
                    ]
                ),
                sample_ids=[row["sample_id"] for row in rows],
                image_paths=[row.get("image", "") for row in rows],
                group_ids=[
                    row.get("group_id", "") or row["sample_id"]
                    for row in rows
                ],
            )
        )
    if not outputs:
        raise FileNotFoundError(f"No fold_*/predictions.csv files found under {root}")
    return outputs

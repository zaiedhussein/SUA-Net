from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Uniform-bin ECE using malignant-class probabilities."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    true = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    if len(true) != len(probability):
        raise ValueError("y_true and y_prob must have equal lengths")
    if len(true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probability, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        accuracy = float(true[mask].mean())
        confidence = float(probability[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return float(ece)


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: Sequence[float]
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    metrics: dict[str, float] = {
        "n_samples": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "recall": float(recall_score(true, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "kappa": float(cohen_kappa_score(true, predicted)),
        "brier": float(brier_score_loss(true, probability)),
        "ece_10": expected_calibration_error(true, probability, n_bins=10),
    }
    if len(np.unique(true)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(true, probability))
        metrics["avg_prec"] = float(average_precision_score(true, probability))
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        metrics["log_loss"] = float(
            log_loss(true, np.column_stack([1 - clipped, clipped]), labels=[0, 1])
        )
    else:
        metrics["auc_roc"] = float("nan")
        metrics["avg_prec"] = float("nan")
        metrics["log_loss"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    metrics["sensitivity"] = float(tp / (tp + fn)) if tp + fn else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if tn + fp else 0.0
    metrics["npv"] = float(tn / (tn + fn)) if tn + fn else 0.0
    metrics["youden_j"] = metrics["sensitivity"] + metrics["specificity"] - 1.0
    return metrics


def bootstrap_auc_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    true = np.asarray(y_true, dtype=int)
    probability = np.asarray(y_prob, dtype=float)
    generator = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstrap):
        indices = generator.randint(0, len(true), len(true))
        sampled_true = true[indices]
        if len(np.unique(sampled_true)) < 2:
            continue
        scores.append(roc_auc_score(sampled_true, probability[indices]))
    if not scores:
        return 0.0, 1.0
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1.0 - alpha))


def summarize_metrics(
    fold_metrics: Sequence[dict[str, float]],
    keys: Iterable[str] | None = None,
) -> dict[str, dict[str, float]]:
    from scipy.stats import t

    if keys is None:
        keys = sorted(set().union(*(metrics.keys() for metrics in fold_metrics)))
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray(
            [
                metrics[key]
                for metrics in fold_metrics
                if key in metrics and np.isfinite(metrics[key])
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        if values.size > 1:
            half_width = float(t.ppf(0.975, values.size - 1) * std / np.sqrt(values.size))
        else:
            half_width = 0.0
        summary[key] = {
            "mean": mean,
            "std": std,
            "ci_lo": mean - half_width,
            "ci_hi": mean + half_width,
            "n": int(values.size),
        }
    return summary

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

plt.switch_backend("Agg")

PRIMARY_METRICS = [
    "accuracy",
    "auc_roc",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
    "mcc",
]


def _prepare(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _finish(fig, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: Sequence[Mapping[str, float]], path: str | Path) -> None:
    output = _prepare(path)
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Training")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row.get("auc_roc", np.nan) for row in history], label="AUC")
    axes[1].plot(
        epochs,
        [row.get("accuracy", np.nan) for row in history],
        label="Accuracy",
    )
    axes[1].plot(
        epochs,
        [row.get("balanced_accuracy", np.nan) for row in history],
        label="Balanced accuracy",
    )
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1.02))
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _finish(fig, output)


def plot_roc(y_true, y_prob, path: str | Path, title: str = "ROC curve") -> None:
    output = _prepare(path)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title=title,
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _finish(fig, output)


def plot_all_roc(
    results: Mapping[str, object], path: str | Path, title: str = "Cross-dataset ROC curves"
) -> None:
    output = _prepare(path)
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, prediction in results.items():
        fpr, tpr, _ = roc_curve(prediction.y_true, prediction.y_prob)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={score:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black")
    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title=title,
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    _finish(fig, output)


def plot_precision_recall(
    y_true, y_prob, path: str | Path, title: str = "Precision-recall curve"
) -> None:
    output = _prepare(path)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    score = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(recall, precision, linewidth=2, label=f"AUPRC = {score:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    _finish(fig, output)


def plot_confusion(y_true, y_pred, path: str | Path, title: str = "Confusion matrix") -> None:
    output = _prepare(path)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    display = ConfusionMatrixDisplay(matrix, display_labels=["Benign", "Malignant"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    _finish(fig, output)


def plot_metric_summary(summary: Mapping[str, Mapping[str, float]], path: str | Path) -> None:
    output = _prepare(path)
    keys = [key for key in PRIMARY_METRICS if key in summary]
    means = [summary[key]["mean"] for key in keys]
    lower = [summary[key]["mean"] - summary[key]["ci_lo"] for key in keys]
    upper = [summary[key]["ci_hi"] - summary[key]["mean"] for key in keys]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(keys))
    ax.bar(positions, means, yerr=np.asarray([lower, upper]), capsize=4)
    ax.set_xticks(positions, [key.replace("_", " ").title() for key in keys], rotation=25)
    ax.set_ylim(min(0, min(means) - 0.1), 1.05)
    ax.set_ylabel("Mean score with 95% CI")
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, output)


def plot_fold_comparison(fold_metrics: Sequence[Mapping[str, float]], path: str | Path) -> None:
    output = _prepare(path)
    keys = [key for key in PRIMARY_METRICS if all(key in row for row in fold_metrics)]
    positions = np.arange(len(keys))
    width = 0.8 / len(fold_metrics)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for index, row in enumerate(fold_metrics):
        offset = (index - (len(fold_metrics) - 1) / 2) * width
        ax.bar(
            positions + offset,
            [row[key] for key in keys],
            width=width,
            label=f"Fold {index + 1}",
        )
    ax.set_xticks(positions, [key.replace("_", " ").title() for key in keys], rotation=20)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("Score")
    ax.legend(ncol=len(fold_metrics), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, output)


def plot_metric_distribution(
    fold_metrics: Sequence[Mapping[str, float]],
    path: str | Path,
    *,
    kind: str,
) -> None:
    if kind not in {"box", "violin"}:
        raise ValueError("kind must be 'box' or 'violin'")
    output = _prepare(path)
    keys = [key for key in PRIMARY_METRICS if all(key in row for row in fold_metrics)]
    values = [[row[key] for row in fold_metrics] for key in keys]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if kind == "box":
        ax.boxplot(values, tick_labels=[key.replace("_", " ").title() for key in keys])
    else:
        ax.violinplot(values, showmeans=True, showmedians=True)
        ax.set_xticks(
            np.arange(1, len(keys) + 1),
            [key.replace("_", " ").title() for key in keys],
        )
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Fold score")
    ax.set_ylim(-0.1, 1.05)
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, output)


def plot_fold_heatmap(fold_metrics: Sequence[Mapping[str, float]], path: str | Path) -> None:
    output = _prepare(path)
    keys = [key for key in PRIMARY_METRICS if all(key in row for row in fold_metrics)]
    values = np.asarray([[row[key] for key in keys] for row in fold_metrics])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=-0.1, vmax=1)
    ax.set_xticks(range(len(keys)), [key.replace("_", " ").title() for key in keys])
    ax.set_yticks(
        range(len(fold_metrics)), [f"Fold {index + 1}" for index in range(len(fold_metrics))]
    )
    ax.tick_params(axis="x", rotation=25)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Score")
    _finish(fig, output)


def plot_radar(summary: Mapping[str, Mapping[str, float]], path: str | Path) -> None:
    output = _prepare(path)
    keys = [key for key in PRIMARY_METRICS if key in summary and key != "mcc"]
    values = [summary[key]["mean"] for key in keys]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1], [key.replace("_", " ").title() for key in keys])
    ax.set_ylim(0, 1)
    _finish(fig, output)


def plot_calibration(y_true, y_prob, path: str | Path) -> None:
    output = _prepare(path)
    observed, predicted = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(predicted, observed, marker="o", linewidth=2, label="SUA-Net")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    ax.set(
        xlabel="Mean predicted malignancy probability",
        ylabel="Observed malignant fraction",
        title="Calibration curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    _finish(fig, output)


def plot_dataset_comparison(
    summaries: Mapping[str, Mapping[str, Mapping[str, float]]], path: str | Path
) -> None:
    output = _prepare(path)
    datasets = list(summaries)
    keys = [
        key for key in PRIMARY_METRICS if all(key in summaries[dataset] for dataset in datasets)
    ]
    positions = np.arange(len(keys))
    width = 0.8 / len(datasets)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for index, dataset in enumerate(datasets):
        offset = (index - (len(datasets) - 1) / 2) * width
        ax.bar(
            positions + offset,
            [summaries[dataset][key]["mean"] for key in keys],
            width=width,
            label=dataset,
        )
    ax.set_xticks(positions, [key.replace("_", " ").title() for key in keys], rotation=20)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("Cross-validation mean")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, output)


def plot_ablation_results(
    summaries: Mapping[str, Mapping[str, Mapping[str, float]]], path: str | Path
) -> None:
    output = _prepare(path)
    metrics = ["accuracy", "auc_roc", "f1", "sensitivity", "specificity", "mcc"]
    variants = list(summaries)
    positions = np.arange(len(metrics))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(15, 6))
    for index, variant in enumerate(variants):
        means = [summaries[variant].get(metric, {}).get("mean", np.nan) for metric in metrics]
        errors = [summaries[variant].get(metric, {}).get("std", 0.0) for metric in metrics]
        offsets = positions + (index - (len(variants) - 1) / 2) * width
        ax.bar(offsets, means, width=width, yerr=errors, capsize=2, label=variant)
    ax.set_xticks(positions, [metric.replace("_", " ").title() for metric in metrics])
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Cross-validation mean ± SD")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, output)


def plot_ablation_delta(
    summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    path: str | Path,
    reference: str = "Full Model",
) -> None:
    if reference not in summaries:
        return
    output = _prepare(path)
    variants = [variant for variant in summaries if variant != reference]
    metrics = ["accuracy", "auc_roc", "f1", "sensitivity", "specificity", "mcc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=True)
    for index, metric in enumerate(metrics):
        reference_value = summaries[reference].get(metric, {}).get("mean", np.nan)
        deltas = [
            summaries[variant].get(metric, {}).get("mean", np.nan) - reference_value
            for variant in variants
        ]
        axes[index].barh(
            range(len(variants)),
            deltas,
            color=["#2e7d32" if delta >= 0 else "#c62828" for delta in deltas],
        )
        axes[index].axvline(0, color="black", linewidth=0.8)
        axes[index].set_title(metric.replace("_", " ").title())
        if index == 0:
            axes[index].set_yticks(range(len(variants)), variants)
    _finish(fig, output)


def plot_ablation_heatmap(
    summaries: Mapping[str, Mapping[str, Mapping[str, float]]], path: str | Path
) -> None:
    output = _prepare(path)
    variants = list(summaries)
    metrics = ["accuracy", "auc_roc", "f1", "sensitivity", "specificity", "precision", "mcc"]
    values = np.asarray(
        [
            [summaries[variant].get(metric, {}).get("mean", np.nan) for metric in metrics]
            for variant in variants
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=-0.1, vmax=1)
    ax.set_xticks(range(len(metrics)), [metric.replace("_", " ").title() for metric in metrics])
    ax.set_yticks(range(len(variants)), variants)
    ax.tick_params(axis="x", rotation=25)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Cross-validation mean")
    _finish(fig, output)


def plot_ablation_effect_sizes(
    statistics: Mapping, path: str | Path, metric: str = "auc_roc"
) -> None:
    comparisons = statistics.get("metrics", {}).get(metric, {})
    if not comparisons:
        return
    output = _prepare(path)
    variants = list(comparisons)
    effects = [comparisons[variant]["paired_cohens_d"] for variant in variants]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(variants))))
    positions = np.arange(len(variants))
    ax.barh(positions, effects)
    ax.axvline(0, linewidth=1, color="black")
    for threshold in (0.2, 0.5, 0.8):
        ax.axvline(threshold, linestyle="--", linewidth=0.8)
        ax.axvline(-threshold, linestyle="--", linewidth=0.8)
    ax.set_yticks(positions, variants)
    ax.set_xlabel(f"Paired Cohen's d for {metric} (Full Model − variant)")
    ax.grid(axis="x", alpha=0.3)
    _finish(fig, output)


def plot_significance_matrix(
    statistics: Mapping, path: str | Path, metric: str = "auc_roc"
) -> None:
    pairs = statistics.get("all_pairwise", {}).get(metric, {})
    variants = sorted({variant for pair in pairs for variant in pair.split(" vs ", maxsplit=1)})
    if not variants:
        return
    output = _prepare(path)
    values = np.ones((len(variants), len(variants)))
    for pair, result in pairs.items():
        first, second = pair.split(" vs ", maxsplit=1)
        row, column = variants.index(first), variants.index(second)
        values[row, column] = values[column, row] = result["wilcoxon_p_bonferroni"]
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(variants)), variants, rotation=45, ha="right")
    ax.set_yticks(range(len(variants)), variants)
    for row in range(len(variants)):
        for column in range(len(variants)):
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Bonferroni-adjusted Wilcoxon p")
    _finish(fig, output)


def plot_auc_matrix(matrix: Mapping[str, Mapping[str, float]], path: str | Path) -> None:
    output = _prepare(path)
    datasets = sorted({name for pair in matrix for name in pair.split("->")})
    values = np.full((len(datasets), len(datasets)), np.nan)
    for pair, metrics in matrix.items():
        train_name, test_name = pair.split("->")
        values[datasets.index(train_name), datasets.index(test_name)] = metrics.get(
            "auc_roc", np.nan
        )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, vmin=0.5, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_yticks(range(len(datasets)), datasets)
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Training dataset")
    for row in range(len(datasets)):
        for column in range(len(datasets)):
            if np.isfinite(values[row, column]):
                ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="AUC-ROC")
    _finish(fig, output)


def plot_generalization_gaps(
    within_auc: Mapping[str, float],
    cross_metrics: Mapping[str, Mapping[str, float]],
    path: str | Path,
) -> None:
    output = _prepare(path)
    pairs = list(cross_metrics)
    gaps = [
        within_auc[pair.split("->")[0]] - cross_metrics[pair].get("auc_roc", np.nan)
        for pair in pairs
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(pairs))
    ax.bar(
        positions,
        gaps,
        color=[
            "#2e7d32" if abs(gap) < 0.05 else "#f9a825" if abs(gap) < 0.1 else "#c62828"
            for gap in gaps
        ],
    )
    ax.axhline(0, linewidth=1, color="black")
    ax.set_xticks(positions, pairs, rotation=35, ha="right")
    ax.set_ylabel("Within-dataset AUC − cross-dataset AUC")
    ax.grid(axis="y", alpha=0.3)
    _finish(fig, output)


def plot_uncertainty_distributions(results: Mapping[str, object], path: str | Path) -> None:
    available = [
        (name, output)
        for name, output in results.items()
        if output.uncertainty is not None and len(output.uncertainty)
    ]
    if not available:
        return
    output_path = _prepare(path)
    columns = min(3, len(available))
    rows = int(np.ceil(len(available) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    for axis, (name, prediction) in zip(axes.flat, available, strict=False):
        correct = prediction.uncertainty[prediction.y_true == prediction.y_pred]
        incorrect = prediction.uncertainty[prediction.y_true != prediction.y_pred]
        groups = [correct]
        labels = ["Correct"]
        if len(incorrect):
            groups.append(incorrect)
            labels.append("Incorrect")
        axis.violinplot(groups, showmeans=True, showmedians=True)
        axis.set_xticks(range(1, len(labels) + 1), labels)
        axis.set_ylabel("Predictive uncertainty")
        axis.set_title(name)
    for axis in axes.flat[len(available) :]:
        axis.set_visible(False)
    _finish(fig, output_path)

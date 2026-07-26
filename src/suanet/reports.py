from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _write(text: str, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")
    return output


def write_training_report(
    *,
    dataset_name: str,
    fold_metrics: Sequence[Mapping[str, float]],
    summary: Mapping[str, Mapping[str, float]],
    path: str | Path,
) -> Path:
    lines = [
        "SUA-Net cross-validation report",
        f"Dataset: {dataset_name}",
        f"Folds: {len(fold_metrics)}",
        "",
        "Aggregate metrics (mean, SD, 95% t-interval)",
        "-" * 70,
    ]
    for metric in (
        "accuracy",
        "auc_roc",
        "f1",
        "sensitivity",
        "specificity",
        "precision",
        "mcc",
        "kappa",
        "avg_prec",
        "ece_10",
        "brier",
    ):
        if metric not in summary:
            continue
        values = summary[metric]
        lines.append(
            f"{metric:<20} {values['mean']:.4f} ± {values['std']:.4f} "
            f"[{values['ci_lo']:.4f}, {values['ci_hi']:.4f}]"
        )
    lines.extend(["", "Fold-level primary metrics", "-" * 70])
    for index, row in enumerate(fold_metrics, start=1):
        lines.append(
            f"Fold {index}: accuracy={row.get('accuracy', float('nan')):.4f}, "
            f"AUC={row.get('auc_roc', float('nan')):.4f}, "
            f"F1={row.get('f1', float('nan')):.4f}, "
            f"MCC={row.get('mcc', float('nan')):.4f}"
        )
    mc_aucs = [float(row["mc_auc"]) for row in fold_metrics if "mc_auc" in row]
    if mc_aucs:
        lines.extend(
            [
                "",
                f"MC-dropout AUC: {np.mean(mc_aucs):.4f} ± {np.std(mc_aucs, ddof=1):.4f}",
                "MC-dropout metrics are auxiliary and do not replace the primary endpoint.",
            ]
        )
    return _write("\n".join(lines), path)


def write_ablation_report(
    *,
    summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    parameter_counts: Mapping[str, Mapping[str, int]],
    statistics: Mapping,
    path: str | Path,
) -> Path:
    lines = [
        "SUA-Net ablation report",
        "",
        "Variant results",
        "-" * 88,
        f"{'Variant':<24} {'Parameters':>14} {'Accuracy':>12} {'AUC':>12} {'F1':>12} {'MCC':>12}",
    ]
    for variant, summary in summaries.items():
        parameters = parameter_counts.get(variant, {}).get("trainable", 0)
        lines.append(
            f"{variant:<24} {parameters:>14,} "
            f"{summary.get('accuracy', {}).get('mean', float('nan')):>12.4f} "
            f"{summary.get('auc_roc', {}).get('mean', float('nan')):>12.4f} "
            f"{summary.get('f1', {}).get('mean', float('nan')):>12.4f} "
            f"{summary.get('mcc', {}).get('mean', float('nan')):>12.4f}"
        )

    lines.extend(["", "Paired comparisons against Full Model", "-" * 88])
    for metric in ("auc_roc", "accuracy", "f1", "mcc"):
        comparisons = statistics.get("metrics", {}).get(metric, {})
        if not comparisons:
            continue
        friedman = statistics.get("friedman", {}).get(metric, {})
        lines.append(f"{metric}: Friedman p={friedman.get('p_value', float('nan')):.6g}")
        for variant, values in comparisons.items():
            lines.append(
                f"  {variant:<22} Δ={values['mean_difference']:+.4f}, "
                f"d={values['paired_cohens_d']:+.3f} ({values['effect_size']}), "
                f"Wilcoxon p_adj={values['wilcoxon_p_bonferroni']:.6g}"
            )
    return _write("\n".join(lines), path)


def generalization_latex(
    *,
    dataset_names: Sequence[str],
    cross_metrics: Mapping[str, Mapping[str, float]],
    within_auc: Mapping[str, float],
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-dataset generalization of SUA-Net. Rows are training datasets and "
        r"columns are test datasets. Diagonal values are within-dataset five-fold means.}",
        r"\label{tab:suanet_generalization}",
        r"\begin{tabular}{l" + "cc" * len(dataset_names) + "}",
        r"\hline",
    ]
    header = r"Train $\rightarrow$ Test"
    for dataset in dataset_names:
        header += rf" & \multicolumn{{2}}{{c}}{{{dataset}}}"
    lines.append(header + r" \\")
    lines.append("".join(r" & AUC & $\Delta$" for _ in dataset_names) + r" \\")
    lines.append(r"\hline")
    for source in dataset_names:
        row = source
        baseline = float(within_auc.get(source, 0.0))
        for target in dataset_names:
            if source == target:
                row += rf" & \textbf{{{baseline:.3f}}} & --"
            else:
                auc = float(cross_metrics.get(f"{source}->{target}", {}).get("auc_roc", 0.0))
                gap = baseline - auc if baseline > 0 else 0.0
                row += rf" & {auc:.3f} & {gap:+.3f}"
        lines.append(row + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])

    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Cross-dataset generalization. $\Delta$ values denote within-dataset source performance minus external-test performance.}",
            r"\label{tab:suanet_generalization_detail}",
            r"\begin{tabular}{llcccccccccc}",
            r"\hline",
            r"Train & Test & Acc & AUC & F1 & Sens & Spec & MCC & $\Delta$AUC & $\Delta$F1 & $\Delta$MCC & $\Delta$Sens \\",
            r"\hline",
        ]
    )
    for pair, metrics in cross_metrics.items():
        source, target = pair.split("->")
        lines.append(
            f"{source} & {target}"
            f" & {metrics.get('accuracy', 0):.3f}"
            f" & {metrics.get('auc_roc', 0):.3f}"
            f" & {metrics.get('f1', 0):.3f}"
            f" & {metrics.get('sensitivity', 0):.3f}"
            f" & {metrics.get('specificity', 0):.3f}"
            f" & {metrics.get('mcc', 0):.3f}"
            f" & {metrics.get('delta_auc', 0):+.3f}"
            f" & {metrics.get('delta_f1', 0):+.3f}"
            f" & {metrics.get('delta_mcc', 0):+.3f}"
            f" & {metrics.get('delta_sensitivity', 0):+.3f}"
            r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)

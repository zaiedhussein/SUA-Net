from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from scipy.stats import chi2, friedmanchisquare, ttest_rel, wilcoxon

from .metrics import bootstrap_auc_ci


def paired_cohens_d(reference: Sequence[float], comparison: Sequence[float]) -> float:
    """Standardized mean of paired fold-wise differences."""
    differences = np.asarray(reference, dtype=float) - np.asarray(comparison, dtype=float)
    if differences.size < 2:
        return 0.0
    standard_deviation = differences.std(ddof=1)
    if standard_deviation < 1e-12:
        return 0.0
    return float(differences.mean() / standard_deviation)


def effect_size_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def bonferroni_correction(p_values: Mapping[str, float]) -> dict[str, float]:
    tests = len(p_values)
    if tests == 0:
        return {}
    return {name: min(float(value) * tests, 1.0) for name, value in p_values.items()}


def _wilcoxon_p(reference: Sequence[float], comparison: Sequence[float]) -> float:
    differences = np.asarray(reference, dtype=float) - np.asarray(comparison, dtype=float)
    if differences.size < 2 or np.all(np.abs(differences) < 1e-12):
        return 1.0
    try:
        return float(wilcoxon(reference, comparison, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def _paired_t_p(reference: Sequence[float], comparison: Sequence[float]) -> float:
    if len(reference) < 2:
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ttest_rel(reference, comparison, nan_policy="omit")
    return float(result.pvalue) if np.isfinite(result.pvalue) else 1.0


def mcnemar_test(
    y_true: Sequence[int],
    reference_prediction: Sequence[int],
    comparison_prediction: Sequence[int],
) -> dict[str, float | int]:
    """Continuity-corrected McNemar test for paired per-image decisions."""
    true = np.asarray(y_true, dtype=int)
    reference = np.asarray(reference_prediction, dtype=int)
    comparison = np.asarray(comparison_prediction, dtype=int)
    if not (len(true) == len(reference) == len(comparison)):
        raise ValueError("McNemar inputs must have equal lengths")
    reference_wrong_comparison_right = int(np.sum((reference != true) & (comparison == true)))
    reference_right_comparison_wrong = int(np.sum((reference == true) & (comparison != true)))
    discordant = reference_wrong_comparison_right + reference_right_comparison_wrong
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float(
            (abs(reference_wrong_comparison_right - reference_right_comparison_wrong) - 1) ** 2
            / discordant
        )
        p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return {
        "reference_wrong_comparison_right": reference_wrong_comparison_right,
        "reference_right_comparison_wrong": reference_right_comparison_wrong,
        "statistic": statistic,
        "p_value": p_value,
    }


def _pooled_prediction_arrays(outputs: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not outputs:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    return (
        np.concatenate([np.asarray(output.y_true, dtype=int) for output in outputs]),
        np.concatenate([np.asarray(output.y_pred, dtype=int) for output in outputs]),
        np.concatenate([np.asarray(output.y_prob, dtype=float) for output in outputs]),
    )


def ablation_statistics(
    all_fold_metrics: Mapping[str, Sequence[Mapping[str, float]]],
    prediction_outputs: Mapping[str, Sequence] | None = None,
    reference_name: str = "Full Model",
    metrics: Sequence[str] = ("auc_roc", "accuracy", "f1", "mcc"),
) -> dict:
    """Complete paired analysis for fixed-fold ablation experiments."""
    if reference_name not in all_fold_metrics:
        raise KeyError(f"Reference variant '{reference_name}' was not found")
    variants = list(all_fold_metrics)
    output: dict = {
        "reference": reference_name,
        "metrics": {},
        "friedman": {},
        "all_pairwise": {},
        "mcnemar_vs_reference": {},
        "pooled_bootstrap_auc": {},
    }

    for metric in metrics:
        fold_scores = {
            variant: [float(row[metric]) for row in rows]
            for variant, rows in all_fold_metrics.items()
        }
        lengths = {variant: len(values) for variant, values in fold_scores.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Ablation variants have different fold counts: {lengths}")

        groups = [fold_scores[variant] for variant in variants]
        if len(groups) >= 3 and len(groups[0]) >= 2:
            statistic, p_value = friedmanchisquare(*groups)
            output["friedman"][metric] = {
                "statistic": float(statistic) if np.isfinite(statistic) else 0.0,
                "p_value": float(p_value) if np.isfinite(p_value) else 1.0,
            }

        reference_values = fold_scores[reference_name]
        reference_comparisons: dict[str, dict[str, float | str]] = {}
        for variant in variants:
            if variant == reference_name:
                continue
            comparison_values = fold_scores[variant]
            effect = paired_cohens_d(reference_values, comparison_values)
            reference_comparisons[variant] = {
                "reference_mean": float(np.mean(reference_values)),
                "comparison_mean": float(np.mean(comparison_values)),
                "mean_difference": float(np.mean(reference_values) - np.mean(comparison_values)),
                "paired_cohens_d": effect,
                "effect_size": effect_size_label(effect),
                "paired_t_p_raw": _paired_t_p(reference_values, comparison_values),
                "wilcoxon_p_raw": _wilcoxon_p(reference_values, comparison_values),
            }
        corrected_t = bonferroni_correction(
            {
                variant: float(values["paired_t_p_raw"])
                for variant, values in reference_comparisons.items()
            }
        )
        corrected_w = bonferroni_correction(
            {
                variant: float(values["wilcoxon_p_raw"])
                for variant, values in reference_comparisons.items()
            }
        )
        for variant, values in reference_comparisons.items():
            values["paired_t_p_bonferroni"] = corrected_t[variant]
            values["wilcoxon_p_bonferroni"] = corrected_w[variant]
        output["metrics"][metric] = reference_comparisons

        pairwise: dict[str, dict[str, float | str]] = {}
        for first, second in combinations(variants, 2):
            first_values = fold_scores[first]
            second_values = fold_scores[second]
            effect = paired_cohens_d(first_values, second_values)
            key = f"{first} vs {second}"
            pairwise[key] = {
                "paired_t_p_raw": _paired_t_p(first_values, second_values),
                "wilcoxon_p_raw": _wilcoxon_p(first_values, second_values),
                "paired_cohens_d": effect,
                "effect_size": effect_size_label(effect),
            }
        corrected_pairwise = bonferroni_correction(
            {key: float(values["wilcoxon_p_raw"]) for key, values in pairwise.items()}
        )
        for key, values in pairwise.items():
            values["wilcoxon_p_bonferroni"] = corrected_pairwise[key]
        output["all_pairwise"][metric] = pairwise

    if prediction_outputs:
        pooled = {
            variant: _pooled_prediction_arrays(outputs)
            for variant, outputs in prediction_outputs.items()
        }
        reference_true, reference_pred, _ = pooled[reference_name]
        for variant, (true, prediction, _) in pooled.items():
            if variant == reference_name:
                continue
            if not np.array_equal(reference_true, true):
                raise ValueError(
                    f"Prediction order differs between '{reference_name}' and '{variant}'"
                )
            output["mcnemar_vs_reference"][variant] = mcnemar_test(
                reference_true, reference_pred, prediction
            )
        for variant, (true, _, probability) in pooled.items():
            low, high = bootstrap_auc_ci(true, probability)
            output["pooled_bootstrap_auc"][variant] = {
                "ci_lo": low,
                "ci_hi": high,
                "n_samples": int(len(true)),
            }

    return output

import numpy as np
import pytest

from suanet.metrics import bootstrap_auc_ci, classification_metrics, summarize_metrics


def test_metrics_are_computed():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.7, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = classification_metrics(y_true, y_pred, y_prob)
    assert metrics["auc_roc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_bootstrap_auc_interval_is_ordered():
    low, high = bootstrap_auc_ci([0, 0, 1, 1], [0.1, 0.3, 0.8, 0.9], n_bootstrap=100)
    assert 0 <= low <= high <= 1


def test_summary_contains_mean_and_ci():
    summary = summarize_metrics([{"auc_roc": 0.8}, {"auc_roc": 0.9}], ["auc_roc"])
    assert summary["auc_roc"]["mean"] == pytest.approx(0.85)
    assert summary["auc_roc"]["ci_lo"] <= 0.85 <= summary["auc_roc"]["ci_hi"]

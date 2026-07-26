from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import suanet.experiments as experiments
from suanet.engine import PredictionOutput


def test_ensemble_averages_fold_mc_predictions_and_reports_fold_uncertainty(
    monkeypatch,
):
    outputs = {
        "fold-a": PredictionOutput(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_prob=np.array([0.2, 0.8]),
            sample_ids=["a", "b"],
        ),
        "fold-b": PredictionOutput(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            y_prob=np.array([0.4, 0.6]),
            sample_ids=["a", "b"],
        ),
    }
    observed_mc_samples = []

    def fake_predict(model, *_args, mc_samples=0, **_kwargs):
        observed_mc_samples.append(mc_samples)
        return outputs[model]

    monkeypatch.setattr(experiments, "predict_loader", fake_predict)
    result = experiments.ensemble_predict(
        ["fold-a", "fold-b"],
        loader=[],
        device=torch.device("cpu"),
        mixed_precision=False,
        mc_samples=7,
    )

    assert observed_mc_samples == [7, 7]
    assert result.y_prob == pytest.approx([0.3, 0.7])
    assert result.y_pred.tolist() == [0, 1]
    assert result.uncertainty == pytest.approx([np.sqrt(0.02), np.sqrt(0.02)])


def test_within_auc_is_loaded_from_training_summary(tmp_path):
    model_dir = tmp_path / "BUSI"
    model_dir.mkdir()
    (model_dir / "summary.json").write_text(
        json.dumps({"summary": {"auc_roc": {"mean": 0.9123}}}),
        encoding="utf-8",
    )
    result = experiments._load_within_auc({"BUSI": model_dir}, explicit={})
    assert result == {"BUSI": pytest.approx(0.9123)}


def test_explicit_within_auc_overrides_summary(tmp_path):
    model_dir = tmp_path / "BUSI"
    model_dir.mkdir()
    (model_dir / "summary.json").write_text(
        json.dumps({"summary": {"auc_roc": {"mean": 0.5}}}),
        encoding="utf-8",
    )
    result = experiments._load_within_auc({"BUSI": model_dir}, explicit={"BUSI": 0.9})
    assert result == {"BUSI": pytest.approx(0.9)}

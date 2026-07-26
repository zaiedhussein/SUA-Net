from __future__ import annotations

import numpy as np
import torch

import suanet.engine as engine
from suanet.config import ExperimentConfig, apply_overrides
from suanet.engine import PredictionOutput


def test_final_model_default_monitor_and_ablation_override():
    config = ExperimentConfig()
    config.dataset.root = "/data/busi"
    config.validate()
    assert config.training.monitor == "accuracy"

    apply_overrides(
        config,
        {
            "training": {
                "monitor": "auc_roc",
                "use_mc_dropout": False,
            }
        },
    )
    assert config.training.monitor == "auc_roc"
    assert config.training.use_mc_dropout is False


def test_fit_fold_keeps_primary_and_mc_dropout_endpoints_separate(monkeypatch, tmp_path):
    primary = PredictionOutput(
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        y_prob=np.array([0.1, 0.9]),
        sample_ids=["a", "b"],
        image_paths=["a.png", "b.png"],
    )
    mc = PredictionOutput(
        y_true=np.array([0, 1]),
        y_pred=np.array([1, 0]),
        y_prob=np.array([0.9, 0.1]),
        uncertainty=np.array([0.2, 0.3]),
        sample_ids=["a", "b"],
        image_paths=["a.png", "b.png"],
    )
    calls = []

    def fake_predict(*_args, mc_samples=0, **_kwargs):
        calls.append(mc_samples)
        return mc if mc_samples else primary

    monkeypatch.setattr(engine, "train_one_epoch", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(engine, "evaluate_loss", lambda *_args, **_kwargs: 0.3)
    monkeypatch.setattr(engine, "predict_loader", fake_predict)
    monkeypatch.setattr(engine, "bootstrap_auc_ci", lambda *_args, **_kwargs: (0.5, 1.0))

    model = torch.nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = engine.fit_fold(
        model=model,
        train_loader=[],
        val_loader=[],
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        epochs=1,
        accumulation=1,
        grad_clip=1.0,
        mixed_precision=False,
        early_stopping=1,
        monitor="accuracy",
        monitor_mode="max",
        checkpoint_path=tmp_path / "best_model.pth",
        checkpoint_metadata={"model": "test"},
        use_tta=False,
        use_mc_dropout=True,
        mc_dropout_samples=5,
    )

    assert calls == [0, 0, 5]
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["mc_accuracy"] == 0.0
    assert np.array_equal(result.predictions.y_prob, primary.y_prob)
    assert np.array_equal(result.predictions.mc_prob, mc.y_prob)
    assert np.array_equal(result.predictions.uncertainty, mc.uncertainty)

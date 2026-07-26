from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from suanet.analysis import paired_mc_across_datasets, threshold_operating_points
from suanet.config import load_experiment_config
from suanet.data import scan_busbra
from suanet.engine import PredictionOutput
from suanet.metrics import expected_calibration_error
from suanet.model import MultiGranularityPooling, SpeckleVarianceAttention, SUANet


def test_mgp_uses_population_standard_deviation():
    features = torch.tensor([[[[0.0, 2.0]]]])
    pooled = MultiGranularityPooling.pool(features)
    assert pooled.tolist() == [[1.0, 2.0, 1.0]]


@pytest.mark.parametrize("attention_mode", ["joint", "channel_only", "spatial_only"])
@pytest.mark.parametrize("variance_mode", ["channel_mean", "channel_preserving"])
def test_design_ablation_sva_modes_forward(attention_mode, variance_mode):
    module = SpeckleVarianceAttention(
        channels=8,
        attention_mode=attention_mode,
        variance_mode=variance_mode,
    )
    inputs = torch.randn(2, 8, 7, 7)
    assert module(inputs).shape == inputs.shape


def test_mamba_backbone_adapter_preserves_channel_axis():
    model = SUANet(encoder_name="mambaout_small", pretrained=False)
    features = model.extract_backbone_feature_map(torch.zeros(1, 3, 224, 224))
    assert features.shape == (1, 576, 7, 7)


def test_full_efficientnet_parameter_count_matches_manuscript():
    model = SUANet(pretrained=False)
    assert model.parameter_counts() == {
        "trainable": 12_207_159,
        "total": 12_225_321,
    }


def test_ece_uses_ten_uniform_bins():
    value = expected_calibration_error([0, 1], [0.1, 0.8], n_bins=10)
    assert value == pytest.approx(0.15)


def test_threshold_analysis_includes_all_manuscript_rules():
    rows = threshold_operating_points(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
    )
    rules = " / ".join(str(row["rule"]) for row in rows)
    assert "Default" in rules
    assert "Max MCC" in rules
    assert "Max Youden" in rules
    assert "Sensitivity >= 0.90" in rules


def test_paired_mc_analysis_clusters_and_holm_corrects():
    output = PredictionOutput(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.2, 0.8, 0.9]),
        mc_prob=np.array([0.2, 0.3, 0.7, 0.8]),
        group_ids=["p1", "p1", "p2", "p2"],
    )
    result = paired_mc_across_datasets(
        {"BUSBRA": [output]},
        n_bootstrap=20,
        seed=42,
    )
    dataset = result["datasets"]["BUSBRA"]
    assert dataset["n_resampling_units"] == 2
    assert "p_holm" in dataset["metrics"]["auc_roc"]
    assert "p_holm" in dataset["mcnemar"]


def test_busbra_case_metadata_drives_groups(tmp_path):
    images = tmp_path / "Images"
    images.mkdir()
    for name in ("bus_0001-1.png", "bus_0001-2.png", "bus_0002-1.png"):
        Image.new("L", (4, 4), color=32).save(images / name)
    with (tmp_path / "bus_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Pathology", "Case"])
        writer.writeheader()
        writer.writerow({"ID": "bus_0001", "Pathology": "benign", "Case": "case-a"})
        writer.writerow({"ID": "bus_0002", "Pathology": "malignant", "Case": "case-b"})
    samples = scan_busbra(tmp_path)
    assert [sample["group_id"] for sample in samples] == ["case-a", "case-a", "case-b"]
    assert all(sample["group_verified"] for sample in samples)


def test_busbra_public_config_enforces_patient_wise_protocol():
    config = load_experiment_config("configs/train/busbra.yaml")
    assert config.dataset.split_strategy == "stratified_group"
    assert config.dataset.require_verified_groups is True

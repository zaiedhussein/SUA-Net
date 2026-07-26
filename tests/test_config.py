from pathlib import Path

import pytest

from suanet.config import load_experiment_config


def test_configuration_loading(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment_name: test
dataset:
  name: BUSI
  root: /data/busi
model:
  encoder_name: scratch
  pretrained: false
training:
  k_folds: 3
runtime:
  output_dir: results/test
""",
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)
    assert config.dataset.name == "BUSI"
    assert config.model.encoder_name == "scratch"
    assert config.training.k_folds == 3


def test_unknown_configuration_key_is_rejected(tmp_path: Path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
dataset:
  name: BUSI
  root: /data/busi
unknown_section: true
""",
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        load_experiment_config(config_path)

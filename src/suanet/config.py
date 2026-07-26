from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "BUSI"
    root: str = ""
    split_strategy: str = "stratified_image"
    classes: list[str] = field(default_factory=lambda: ["benign", "malignant"])
    require_verified_groups: bool = False


@dataclass
class ModelConfig:
    encoder_name: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.4
    freeze_blocks: int = 2
    sva_kernel: int = 7
    sva_reduction: int = 8
    dla_dilation: int = 3
    use_sva: bool = True
    use_dla: bool = True
    use_mgp: bool = True
    sva_attention_mode: str = "joint"
    sva_variance_mode: str = "channel_mean"
    mgp_statistics: list[str] = field(default_factory=lambda: ["avg", "max", "std"])


@dataclass
class TrainingConfig:
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    early_stopping: int = 20
    accumulation: int = 2
    k_folds: int = 5
    seed: int = 42
    monitor: str = "accuracy"
    monitor_mode: str = "max"
    strong_augmentation: bool = True
    mixed_precision: bool = True
    use_tta: bool = False
    use_mc_dropout: bool = True
    mc_dropout_samples: int = 20
    imbalance_strategy: str = "baseline"


@dataclass
class RuntimeConfig:
    output_dir: str = "results/run"
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    deterministic: bool = True
    save_predictions: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "suanet"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dataset.name.upper().replace("-", "").replace("_", "") not in {
            "BUSI",
            "BUSBRA",
            "BUSUCLM",
            "UCLM",
        }:
            raise ValueError("dataset.name must be BUSI, BUSBRA, or BUS-UCLM")
        if not self.dataset.root:
            raise ValueError("dataset.root must be configured")
        if self.dataset.split_strategy not in {"stratified_image", "stratified_group"}:
            raise ValueError(
                "dataset.split_strategy must be 'stratified_image' or 'stratified_group'"
            )
        if self.dataset.require_verified_groups and self.dataset.split_strategy != "stratified_group":
            raise ValueError("require_verified_groups requires stratified_group splitting")
        if self.model.num_classes != 2:
            raise ValueError("SUA-Net currently supports binary classification (num_classes=2)")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.model.freeze_blocks < 0:
            raise ValueError("model.freeze_blocks must be non-negative")
        if self.model.sva_kernel < 1 or self.model.sva_kernel % 2 == 0:
            raise ValueError("model.sva_kernel must be a positive odd integer")
        if self.model.sva_reduction < 1:
            raise ValueError("model.sva_reduction must be at least 1")
        if self.model.dla_dilation < 1:
            raise ValueError("model.dla_dilation must be at least 1")
        if self.model.sva_attention_mode not in {"joint", "channel_only", "spatial_only"}:
            raise ValueError(
                "model.sva_attention_mode must be joint, channel_only, or spatial_only"
            )
        if self.model.sva_variance_mode not in {"channel_mean", "channel_preserving"}:
            raise ValueError(
                "model.sva_variance_mode must be channel_mean or channel_preserving"
            )
        valid_statistics = {"avg", "max", "std", "skew"}
        if (
            not self.model.mgp_statistics
            or len(set(self.model.mgp_statistics)) != len(self.model.mgp_statistics)
            or not set(self.model.mgp_statistics) <= valid_statistics
        ):
            raise ValueError(
                "model.mgp_statistics must contain unique avg, max, std, and/or skew entries"
            )
        if self.training.img_size < 16:
            raise ValueError("training.img_size must be at least 16")
        if self.training.batch_size < 1 or self.training.epochs < 1:
            raise ValueError("training.batch_size and training.epochs must be positive")
        if self.training.lr <= 0 or self.training.weight_decay < 0:
            raise ValueError("training.lr must be positive and weight_decay non-negative")
        if self.training.grad_clip <= 0:
            raise ValueError("training.grad_clip must be positive")
        if self.training.early_stopping < 1:
            raise ValueError("training.early_stopping must be at least 1")
        if self.training.accumulation < 1:
            raise ValueError("training.accumulation must be at least 1")
        if self.training.k_folds < 2:
            raise ValueError("training.k_folds must be at least 2")
        if self.training.monitor_mode not in {"max", "min"}:
            raise ValueError("training.monitor_mode must be 'max' or 'min'")
        if self.training.imbalance_strategy not in {
            "baseline",
            "weighted_focal",
            "balanced_sampler",
        }:
            raise ValueError(
                "training.imbalance_strategy must be baseline, weighted_focal, "
                "or balanced_sampler"
            )
        if self.training.use_mc_dropout and self.training.mc_dropout_samples < 2:
            raise ValueError(
                "training.mc_dropout_samples must be at least 2 when MC-dropout is enabled"
            )
        if self.runtime.num_workers < 0:
            raise ValueError("runtime.num_workers must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: Mapping[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, Mapping):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def apply_overrides(
    config: ExperimentConfig, overrides: Mapping[str, Any] | None
) -> ExperimentConfig:
    if overrides:
        _merge_dataclass(config, overrides)
    config.validate()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = _merge_dataclass(ExperimentConfig(), payload)
    config.validate()
    return config


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload

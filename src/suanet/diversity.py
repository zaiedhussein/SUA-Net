from __future__ import annotations

import csv
import platform
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import BreastUltrasoundDataset, build_transforms, scan_dataset
from .utils import resolve_device, save_json, seed_everything


def _native_image_statistics(samples: list[dict]) -> dict[str, float]:
    means = []
    gradient_energies = []
    for sample in tqdm(samples, desc="native image statistics", leave=False):
        image = np.asarray(Image.open(str(sample["image"])).convert("L"), dtype=np.float64) / 255.0
        means.append(float(image.mean()))
        horizontal = np.diff(image, axis=1)
        vertical = np.diff(image, axis=0)
        gradient_energies.append(
            0.5 * (float(np.mean(horizontal**2)) + float(np.mean(vertical**2)))
        )
    return {
        "intensity_mean_sd": float(np.std(means, ddof=1)),
        "gradient_energy": float(np.mean(gradient_energies)),
    }


def _feature_map(output: torch.Tensor, channels: int) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if output.ndim == 4 and output.shape[1] == channels:
        return output
    if output.ndim == 4 and output.shape[-1] == channels:
        return output.permute(0, 3, 1, 2)
    if output.ndim == 3 and output.shape[-1] == channels:
        token_count = output.shape[1]
        side = int(token_count**0.5)
        if side * side != token_count:
            output = output[:, 1:, :]
            token_count -= 1
            side = int(token_count**0.5)
        if side * side != token_count:
            raise RuntimeError(f"Cannot reshape backbone output {tuple(output.shape)}")
        return output.transpose(1, 2).reshape(output.shape[0], channels, side, side)
    raise RuntimeError(f"Unsupported backbone output shape: {tuple(output.shape)}")


def _extract_features(
    samples: list[dict],
    *,
    model: torch.nn.Module,
    feature_channels: int,
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataset = BreastUltrasoundDataset(
        samples,
        img_size,
        transform=build_transforms(img_size, training=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    vectors = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="pretrained features", leave=False):
            output = model(images.to(device, non_blocking=True))
            feature_map = _feature_map(output, feature_channels)
            pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
            vectors.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(vectors, axis=0)


def _bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    generator = np.random.RandomState(seed)
    units = np.unique(groups)
    indices = {unit: np.flatnonzero(groups == unit) for unit in units}
    estimates = []
    for _ in range(n_bootstrap):
        sampled_units = generator.choice(units, size=len(units), replace=True)
        sampled = np.concatenate([indices[unit] for unit in sampled_units])
        estimates.append(float(values[sampled].mean()))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _dataset_diversity(
    features: np.ndarray,
    groups: np.ndarray,
    *,
    max_pairs: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float | int | list[float]], np.ndarray]:
    centroid = features.mean(axis=0)
    distances = np.linalg.norm(features - centroid, axis=1)
    pair_count = min(max_pairs, len(features) * (len(features) - 1) // 2)
    generator = np.random.RandomState(seed)
    first = generator.randint(0, len(features), size=pair_count)
    second = generator.randint(0, len(features), size=pair_count)
    collisions = first == second
    while np.any(collisions):
        second[collisions] = generator.randint(0, len(features), size=int(collisions.sum()))
        collisions = first == second
    pairwise = np.linalg.norm(features[first] - features[second], axis=1)
    centered = features - centroid
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values**2 / max(len(features) - 1, 1)
    positive = eigenvalues[eigenvalues > 1e-15]
    proportions = positive / positive.sum()
    effective_rank = float(np.exp(-np.sum(proportions * np.log(proportions))))
    ci_lo, ci_hi = _bootstrap_mean(
        distances,
        groups,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return (
        {
            "dispersion": float(distances.mean()),
            "dispersion_sd": float(distances.std(ddof=1)),
            "dispersion_ci_95": [ci_lo, ci_hi],
            "pairwise_distance": float(pairwise.mean()),
            "pair_count": int(pair_count),
            "covariance_trace": float(np.sum(centered**2) / max(len(features) - 1, 1)),
            "effective_rank": effective_rank,
        },
        centroid,
    )


def run_diversity_analysis(payload: Mapping) -> dict:
    """Reproduce native-image diversity and pretrained representation-shift analysis."""
    try:
        import timm
    except ImportError as error:  # pragma: no cover
        raise ImportError("timm is required for diversity analysis") from error

    datasets = payload["datasets"]
    output_dir = Path(payload.get("output_dir", "results/diversity"))
    output_dir.mkdir(parents=True, exist_ok=True)
    img_size = int(payload.get("img_size", 224))
    batch_size = int(payload.get("batch_size", 16))
    num_workers = int(payload.get("num_workers", 0))
    seed = int(payload.get("seed", 42))
    max_pairs = int(payload.get("max_pairs", 100000))
    n_bootstrap = int(payload.get("n_bootstrap", 2000))
    device = resolve_device(str(payload.get("device", "auto")))
    seed_everything(seed, True)

    model_name = str(payload.get("encoder_name", "efficientnet_b0"))
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        global_pool="",
    ).to(device)
    channels = int(getattr(model, "num_features", 1280))
    results = {}
    centroids = {}
    for dataset_name, root in datasets.items():
        samples = scan_dataset(root, dataset_name)
        labels = np.asarray([int(sample["label"]) for sample in samples])
        counts = np.bincount(labels, minlength=2)
        proportions = counts / counts.sum()
        nonzero = proportions[proportions > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        verified = all(bool(sample.get("group_verified", False)) for sample in samples)
        groups = np.asarray(
            [
                str(sample.get("group_id", sample["sample_id"]))
                if verified
                else str(sample["sample_id"])
                for sample in samples
            ],
            dtype=object,
        )
        features = _extract_features(
            samples,
            model=model,
            feature_channels=channels,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        diversity, centroid = _dataset_diversity(
            features,
            groups,
            max_pairs=max_pairs,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        native = _native_image_statistics(samples)
        results[dataset_name] = {
            "images": len(samples),
            "benign": int(counts[0]),
            "malignant": int(counts[1]),
            "cases": int(len(np.unique(groups))) if verified else None,
            "class_entropy_bits": entropy,
            **native,
            **diversity,
            "bootstrap_resampling": "patient/group" if verified else "image",
        }
        centroids[dataset_name] = centroid

    shifts = {}
    names = list(datasets)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            shifts[f"{first}--{second}"] = float(
                np.linalg.norm(centroids[first] - centroids[second])
            )
    result = {
        "protocol": {
            "encoder_name": model_name,
            "timm_version": timm.__version__,
            "pretrained_config": getattr(model, "pretrained_cfg", {}),
            "feature_dimension": channels,
            "img_size": img_size,
            "l2_normalized": True,
            "max_pairs": max_pairs,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "sample_size_matching": False,
            "python": platform.python_version(),
            "pytorch": torch.__version__,
        },
        "within_dataset": results,
        "centroid_distances": shifts,
    }
    save_json(result, output_dir / "diversity_results.json")
    with (output_dir / "within_dataset_diversity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        rows = [{"dataset": name, **metrics} for name, metrics in results.items()]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result

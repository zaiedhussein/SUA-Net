from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

plt.switch_backend("Agg")


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[-1]
        self.activations = output

    def _backward_hook(self, _module, _gradient_input, gradient_output) -> None:
        self.gradients = gradient_output[0]

    def generate(self, inputs: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        self.model.eval()
        inputs = inputs.detach().clone().requires_grad_(True)
        logits = self.model(inputs)
        if target_class is None:
            target_class = int(logits.argmax(dim=1)[0])
        self.model.zero_grad(set_to_none=True)
        logits[:, target_class].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap[0, 0].detach().float().cpu().numpy()
        minimum, maximum = float(heatmap.min()), float(heatmap.max())
        if maximum > minimum:
            return (heatmap - minimum) / (maximum - minimum)
        return np.zeros_like(heatmap)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def select_failure_indices(y_true, y_pred, y_prob, max_cases: int = 6) -> list[int]:
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    probability = np.asarray(y_prob)
    false_positive = np.where((true == 0) & (predicted == 1))[0]
    false_negative = np.where((true == 1) & (predicted == 0))[0]
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected:
            selected.append(index)

    if false_positive.size:
        add(int(false_positive[np.argmax(probability[false_positive])]))
        add(int(false_positive[np.argmin(np.abs(probability[false_positive] - 0.5))]))
    if false_negative.size:
        add(int(false_negative[np.argmin(probability[false_negative])]))
        add(int(false_negative[np.argmin(np.abs(probability[false_negative] - 0.5))]))
    all_errors = np.where(true != predicted)[0]
    for index in all_errors[np.argsort(-np.abs(probability[all_errors] - 0.5))]:
        add(int(index))
        if len(selected) >= max_cases:
            break
    return selected[:max_cases]


def select_balanced_indices(samples: Sequence[Mapping], max_cases: int = 8) -> list[int]:
    half = max(1, max_cases // 2)
    benign = [index for index, sample in enumerate(samples) if int(sample["label"]) == 0]
    malignant = [index for index, sample in enumerate(samples) if int(sample["label"]) == 1]
    return (benign[:half] + malignant[:half])[:max_cases]


def generate_gradcam_maps(
    *,
    model: torch.nn.Module,
    dataset,
    indices: Sequence[int],
    device: torch.device,
) -> list[np.ndarray]:
    gradcam = GradCAM(model, model.target_layer())
    maps = []
    try:
        for index in indices:
            tensor, *_ = dataset[index]
            maps.append(gradcam.generate(tensor.unsqueeze(0).to(device)))
    finally:
        gradcam.close()
    return maps


def save_failure_case_figure(
    *,
    model: torch.nn.Module,
    dataset,
    prediction_output,
    indices: Sequence[int],
    device: torch.device,
    path: str | Path,
    dataset_name: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not indices:
        raise RuntimeError("No misclassified cases were available for visualization")

    gradcam = GradCAM(model, model.target_layer())
    fig, axes = plt.subplots(len(indices), 3, figsize=(10, 3.4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    try:
        for row, index in enumerate(indices):
            tensor, label, metadata = dataset[index]
            probability = float(prediction_output.y_prob[index])
            prediction = int(prediction_output.y_pred[index])
            heatmap = gradcam.generate(tensor.unsqueeze(0).to(device), target_class=prediction)

            image = np.asarray(Image.open(metadata["image"]).convert("L"))
            resized = np.asarray(
                Image.fromarray(image).resize((heatmap.shape[1], heatmap.shape[0]))
            )
            axes[row, 0].imshow(image, cmap="gray")
            axes[row, 0].set_title("Original ultrasound")
            axes[row, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            axes[row, 1].set_title("Grad-CAM")
            axes[row, 2].imshow(resized, cmap="gray")
            axes[row, 2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            truth_name = "Malignant" if label == 1 else "Benign"
            predicted_name = "Malignant" if prediction == 1 else "Benign"
            axes[row, 2].set_title(
                f"Truth: {truth_name} | Prediction: {predicted_name}\n"
                f"P(malignant)={probability:.3f}"
            )
            for column in range(3):
                axes[row, column].axis("off")
    finally:
        gradcam.close()

    fig.suptitle(f"Representative failure cases — {dataset_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_high_uncertainty_indices(uncertainty, max_cases: int = 6) -> list[int]:
    values = np.asarray(uncertainty, dtype=float)
    if values.ndim != 1:
        raise ValueError("uncertainty must be one-dimensional")
    finite = np.flatnonzero(np.isfinite(values))
    return [int(index) for index in finite[np.argsort(-values[finite])[:max_cases]]]


def _sva_variance_map(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    if not hasattr(model, "sva") or not hasattr(model.sva, "local_variance"):
        raise RuntimeError("The selected model does not expose an SVA variance map")
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0].detach())

    handle = model.sva.register_forward_pre_hook(capture)
    try:
        model.eval()
        with torch.no_grad():
            model(tensor.unsqueeze(0).to(device))
        if not captured:
            raise RuntimeError("SVA input features were not captured")
        variance = model.sva.local_variance(captured[0]).mean(dim=1, keepdim=True)
        variance = F.interpolate(
            variance,
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        values = variance.float().cpu().numpy()
        minimum, maximum = float(values.min()), float(values.max())
        return (values - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(values)
    finally:
        handle.remove()


def save_uncertainty_case_figure(
    *,
    model: torch.nn.Module,
    dataset,
    prediction_output,
    indices: Sequence[int],
    device: torch.device,
    path: str | Path,
    dataset_name: str,
) -> None:
    """Create the manuscript-style uncertainty, Grad-CAM, and SVA/variance panel."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if prediction_output.uncertainty is None:
        raise ValueError("MC-Dropout uncertainty is required")
    if not indices:
        raise RuntimeError("No high-uncertainty cases were selected")

    gradcam = GradCAM(model, model.target_layer())
    fig, axes = plt.subplots(
        len(indices),
        4,
        figsize=(13, 3.2 * len(indices)),
        squeeze=False,
    )
    try:
        for row, index in enumerate(indices):
            tensor, label, metadata = dataset[index]
            probability = float(
                prediction_output.mc_prob[index]
                if prediction_output.mc_prob is not None
                else prediction_output.y_prob[index]
            )
            prediction = int(probability >= 0.5)
            uncertainty = float(prediction_output.uncertainty[index])
            heatmap = gradcam.generate(
                tensor.unsqueeze(0).to(device),
                target_class=prediction,
            )
            variance = _sva_variance_map(model, tensor, device)
            image = np.asarray(Image.open(metadata["image"]).convert("L"))
            resized = np.asarray(
                Image.fromarray(image).resize((heatmap.shape[1], heatmap.shape[0]))
            )
            axes[row, 0].imshow(image, cmap="gray")
            axes[row, 0].set_title("Ultrasound")
            axes[row, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            axes[row, 1].set_title("Grad-CAM")
            axes[row, 2].imshow(variance, cmap="magma", vmin=0, vmax=1)
            axes[row, 2].set_title("SVA variance")
            axes[row, 3].imshow(resized, cmap="gray")
            axes[row, 3].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            truth_name = "Malignant" if label == 1 else "Benign"
            predicted_name = "Malignant" if prediction == 1 else "Benign"
            axes[row, 3].set_title(
                f"Truth: {truth_name} | Prediction: {predicted_name}\n"
                f"P(malignant)={probability:.3f} | uncertainty={uncertainty:.4f}"
            )
            for column in range(4):
                axes[row, column].axis("off")
    finally:
        gradcam.close()
    fig.suptitle(f"High-uncertainty validation cases — {dataset_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_ablation_gradcam_grid(
    *,
    dataset,
    indices: Sequence[int],
    heatmaps: Mapping[str, Sequence[np.ndarray]],
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    variants = list(heatmaps)
    fig, axes = plt.subplots(
        len(indices),
        len(variants) + 1,
        figsize=(2.6 * (len(variants) + 1), 2.7 * len(indices)),
        squeeze=False,
    )
    for row, index in enumerate(indices):
        _, label, metadata = dataset[index]
        image = np.asarray(Image.open(metadata["image"]).convert("L"))
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_ylabel("Malignant" if label else "Benign")
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title("Original")
        for column, variant in enumerate(variants, start=1):
            heatmap = heatmaps[variant][row]
            resized = np.asarray(
                Image.fromarray(image).resize((heatmap.shape[1], heatmap.shape[0]))
            )
            axes[row, column].imshow(resized, cmap="gray")
            axes[row, column].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(variant, fontsize=8)
    fig.suptitle("SUA-Net ablation Grad-CAM comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)

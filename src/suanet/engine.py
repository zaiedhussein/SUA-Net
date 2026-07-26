from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .metrics import bootstrap_auc_ci, classification_metrics
from .utils import save_json


@dataclass
class PredictionOutput:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    uncertainty: np.ndarray | None = None
    mc_prob: np.ndarray | None = None
    sample_ids: list[str] | None = None
    image_paths: list[str] | None = None
    group_ids: list[str] | None = None


@dataclass
class FoldResult:
    metrics: dict[str, float]
    predictions: PredictionOutput
    history: list[dict[str, float]]
    checkpoint_path: str


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    accumulation: int,
    grad_clip: float,
    mixed_precision: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, mixed_precision):
            logits = model(images)
            loss = criterion(logits, labels) / accumulation

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = (step + 1) % accumulation == 0 or (step + 1) == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu()) * accumulation

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict_batch_tta(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    variants = [
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
        torch.flip(images, dims=[2, 3]),
    ]
    probabilities = [torch.softmax(model(variant), dim=1) for variant in variants]
    return torch.stack(probabilities).mean(dim=0)


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool = True,
    use_tta: bool = False,
    mc_samples: int = 0,
) -> PredictionOutput:
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    predictions_all: list[int] = []
    uncertainties_all: list[float] = []
    sample_ids: list[str] = []
    image_paths: list[str] = []
    group_ids: list[str] = []

    model.eval()
    for batch in tqdm(loader, desc="inference", leave=False):
        images, labels = batch[:2]
        metadata = batch[2] if len(batch) > 2 else None
        images = images.to(device, non_blocking=True)

        if mc_samples > 0:
            model.eval()
            model.enable_mc_dropout()
            draws = []
            for _ in range(mc_samples):
                with _autocast(device, mixed_precision):
                    draws.append(torch.softmax(model(images), dim=1))
            stacked = torch.stack(draws)
            probabilities = stacked.mean(dim=0)
            # Equation in the manuscript uses the population standard deviation
            # and averages dispersion over both output classes.
            uncertainty = stacked.std(dim=0, correction=0).mean(dim=1)
        else:
            model.eval()
            with _autocast(device, mixed_precision):
                if use_tta:
                    probabilities = predict_batch_tta(model, images)
                else:
                    probabilities = torch.softmax(model(images), dim=1)
            uncertainty = None

        predictions = probabilities.argmax(dim=1)
        labels_all.extend(labels.tolist())
        probabilities_all.extend(probabilities[:, 1].float().cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        if uncertainty is not None:
            uncertainties_all.extend(uncertainty.float().cpu().tolist())

        if metadata is not None:
            if isinstance(metadata, dict):
                sample_ids.extend([str(x) for x in metadata.get("sample_id", [])])
                image_paths.extend([str(x) for x in metadata.get("image", [])])
                group_ids.extend([str(x) for x in metadata.get("group_id", [])])
            else:
                for item in metadata:
                    sample_ids.append(str(item.get("sample_id", "")))
                    image_paths.append(str(item.get("image", "")))
                    group_ids.append(str(item.get("group_id", "")))

    return PredictionOutput(
        y_true=np.asarray(labels_all, dtype=int),
        y_pred=np.asarray(predictions_all, dtype=int),
        y_prob=np.asarray(probabilities_all, dtype=float),
        uncertainty=np.asarray(uncertainties_all, dtype=float) if uncertainties_all else None,
        sample_ids=sample_ids or None,
        image_paths=image_paths or None,
        group_ids=group_ids or None,
    )


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="validation", leave=False):
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast(device, mixed_precision):
            losses.append(float(criterion(model(images), labels).cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _is_better(value: float, best: float, mode: str) -> bool:
    return value > best if mode == "max" else value < best


def fit_fold(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    accumulation: int,
    grad_clip: float,
    mixed_precision: bool,
    early_stopping: int,
    monitor: str,
    monitor_mode: str,
    checkpoint_path: str | Path,
    checkpoint_metadata: dict,
    use_tta: bool,
    use_mc_dropout: bool,
    mc_dropout_samples: int,
) -> FoldResult:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_value = -float("inf") if monitor_mode == "max" else float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if mixed_precision and device.type == "cuda"
        else None
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            accumulation,
            grad_clip,
            mixed_precision,
        )
        validation_loss = evaluate_loss(model, val_loader, criterion, device, mixed_precision)
        predictions = predict_loader(
            model,
            val_loader,
            device,
            mixed_precision=mixed_precision,
            use_tta=False,
            mc_samples=0,
        )
        metrics = classification_metrics(predictions.y_true, predictions.y_pred, predictions.y_prob)
        metrics["train_loss"] = train_loss
        metrics["val_loss"] = validation_loss
        metrics["epoch"] = epoch
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(metrics)

        if monitor not in metrics:
            raise KeyError(f"Monitored metric '{monitor}' was not computed")
        monitored_value = float(metrics[monitor])
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
            f"auc={metrics.get('auc_roc', float('nan')):.4f} "
            f"accuracy={metrics['accuracy']:.4f} balanced_accuracy={metrics['balanced_accuracy']:.4f}"
        )

        if not np.isfinite(monitored_value):
            monitored_value = -float("inf") if monitor_mode == "max" else float("inf")
        if not checkpoint_path.exists() or _is_better(monitored_value, best_value, monitor_mode):
            best_value = monitored_value
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "monitor": monitor,
                    "monitor_value": monitored_value,
                    "metadata": checkpoint_metadata,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()
        if epochs_without_improvement >= early_stopping:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    # Preserve the primary endpoint from the final-model workflow: report
    # deterministic (or explicitly enabled TTA) predictions. MC-dropout is
    # evaluated separately and never silently replaces the primary endpoint.
    final_predictions = predict_loader(
        model,
        val_loader,
        device,
        mixed_precision=mixed_precision,
        use_tta=use_tta,
        mc_samples=0,
    )
    final_metrics = classification_metrics(
        final_predictions.y_true,
        final_predictions.y_pred,
        final_predictions.y_prob,
    )
    ci_low, ci_high = bootstrap_auc_ci(final_predictions.y_true, final_predictions.y_prob)
    final_metrics["auc_ci_lo"] = ci_low
    final_metrics["auc_ci_hi"] = ci_high
    if use_mc_dropout:
        mc_predictions = predict_loader(
            model,
            val_loader,
            device,
            mixed_precision=mixed_precision,
            use_tta=False,
            mc_samples=mc_dropout_samples,
        )
        if not np.array_equal(final_predictions.y_true, mc_predictions.y_true):
            raise RuntimeError("MC-dropout and primary inference returned different sample order")
        final_predictions.mc_prob = mc_predictions.y_prob
        final_predictions.uncertainty = mc_predictions.uncertainty
        mc_metrics = classification_metrics(
            mc_predictions.y_true,
            mc_predictions.y_pred,
            mc_predictions.y_prob,
        )
        final_metrics["mc_auc"] = mc_metrics["auc_roc"]
        final_metrics["mc_accuracy"] = mc_metrics["accuracy"]
        if mc_predictions.uncertainty is not None:
            final_metrics["mean_uncertainty"] = float(mc_predictions.uncertainty.mean())
    final_metrics["best_epoch"] = int(checkpoint["epoch"])
    final_metrics["best_monitor_value"] = float(checkpoint["monitor_value"])

    save_json(history, checkpoint_path.parent / "history.json")
    save_json(final_metrics, checkpoint_path.parent / "metrics.json")
    return FoldResult(
        metrics=final_metrics,
        predictions=final_predictions,
        history=history,
        checkpoint_path=str(checkpoint_path),
    )


def load_checkpoint_state(
    path: str | Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload, {"legacy_raw_state_dict": True}
    raise TypeError(f"Unsupported checkpoint format: {path}")

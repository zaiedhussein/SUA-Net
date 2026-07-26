from __future__ import annotations

import csv
import hashlib
import inspect
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:  # pragma: no cover - exercised only in minimal installations
    A = None
    ToTensorV2 = None

Sample = dict[str, object]


def _is_mask_file(path: Path) -> bool:
    return "mask" in path.name.lower()


def scan_busi(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    for class_name, label in (("benign", 0), ("malignant", 1)):
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*.png")):
            if _is_mask_file(image_path):
                continue
            samples.append(
                {
                    "image": str(image_path),
                    "label": label,
                    "sample_id": image_path.stem,
                    "group_id": image_path.stem,
                    "group_source": "image_id",
                    "group_verified": False,
                }
            )
    if not samples:
        raise RuntimeError(f"No BUSI classification images found under {root}")
    return samples


def _normalise_csv_row(row: dict) -> dict[str, str]:
    return {
        str(key).strip().lstrip("\ufeff").lower(): str(value).strip()
        for key, value in row.items()
        if key is not None and value is not None
    }


def _read_busbra_metadata(root: Path) -> dict[str, dict[str, str]]:
    candidates = []
    direct = [root / "bus_data.csv", root.parent / "bus_data.csv"]
    for path in direct:
        if path.is_file() and path not in candidates:
            candidates.append(path)
    if root.exists():
        for path in sorted(root.rglob("*.csv")):
            if path not in candidates:
                candidates.append(path)

    best_records: dict[str, dict[str, str]] = {}
    best_score = (-1, -1)
    for csv_path in candidates:
        current: dict[str, dict[str, str]] = {}
        with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
            for row in csv.DictReader(handle):
                normalized = _normalise_csv_row(row)
                id_key = next(
                    (
                        key
                        for key in ("id", "image", "image_id", "filename", "file", "name")
                        if normalized.get(key)
                    ),
                    None,
                )
                if id_key is None:
                    continue
                record = {
                    "id": normalized[id_key],
                    "pathology": next(
                        (
                            normalized[key].lower()
                            for key in ("pathology", "diagnosis", "class", "label")
                            if normalized.get(key)
                        ),
                        "",
                    ),
                    "case": next(
                        (
                            normalized[key]
                            for key in (
                                "case",
                                "case_id",
                                "patient",
                                "patient_id",
                                "patientid",
                            )
                            if normalized.get(key)
                        ),
                        "",
                    ),
                    "metadata_file": str(csv_path),
                }
                if record["pathology"] not in {"benign", "malignant"}:
                    continue
                aliases = {
                    record["id"],
                    Path(record["id"]).name,
                    Path(record["id"]).stem,
                }
                digits = re.findall(r"\d+", record["id"])
                if digits:
                    aliases.update({digits[-1], str(int(digits[-1]))})
                for alias in aliases:
                    if alias:
                        current[alias.lower()] = record
        score = (
            sum(bool(record["case"]) for record in current.values()),
            len(current),
        )
        if score > best_score:
            best_records = current
            best_score = score
    return best_records

def _lookup_busbra_record(
    records: dict[str, dict[str, str]], image_path: Path
) -> dict[str, str] | None:
    stem = image_path.stem
    aliases = [
        image_path.name,
        stem,
        stem.rsplit("-", maxsplit=1)[0],
        stem.split("-")[0],
    ]
    digits = re.findall(r"\d+", stem)
    if digits:
        aliases.extend([digits[-1], str(int(digits[-1]))])
    return next((records[alias.lower()] for alias in aliases if alias.lower() in records), None)


def _busbra_layout(root: Path) -> tuple[str, Path]:
    image_directories = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and path.name.lower() == "images"
        and any(path.glob("*.png"))
    ]
    if image_directories:
        images = max(image_directories, key=lambda path: len(list(path.glob("*.png"))))
        return "csv", images.parent
    if any(path.is_dir() and path.name.isdigit() for path in root.iterdir()):
        return "birads", root
    birads_roots = [
        path
        for path in root.rglob("*")
        if path.is_dir()
        and any(child.is_dir() and child.name.isdigit() for child in path.iterdir())
    ]
    if birads_roots:
        return "birads", max(
            birads_roots,
            key=lambda path: sum(1 for child in path.rglob("*.png") if child.is_file()),
        )
    raise RuntimeError(f"No supported BUSBRA layout found under {root}")

def scan_busbra(root: str | Path) -> list[Sample]:
    root = Path(root)
    if not root.exists():
        raise RuntimeError(f"BUSBRA root does not exist: {root}")
    records = _read_busbra_metadata(root)
    layout, layout_root = _busbra_layout(root)
    samples: list[Sample] = []
    birads_to_label = {"1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1}

    if layout == "csv":
        image_paths = sorted((layout_root / "Images").glob("*.png"))
        labelled_paths = [(path, None) for path in image_paths]
    else:
        labelled_paths = [
            (path, birads_to_label[folder.name])
            for folder in sorted(
                (
                    path
                    for path in layout_root.iterdir()
                    if path.is_dir() and path.name in birads_to_label
                ),
                key=lambda path: int(path.name),
            )
            for path in sorted(folder.glob("*.png"))
        ]

    for image_path, birads_label in labelled_paths:
        record = _lookup_busbra_record(records, image_path)
        pathology = record["pathology"] if record else ""
        if pathology in {"benign", "malignant"}:
            label = 0 if pathology == "benign" else 1
        elif birads_label is not None:
            label = int(birads_label)
        else:
            continue
        case = record["case"] if record else ""
        fallback_group = image_path.stem.rsplit("-", maxsplit=1)[0]
        samples.append(
            {
                "image": str(image_path),
                "label": label,
                "sample_id": image_path.stem,
                "group_id": case or fallback_group,
                "group_source": "Case" if case else "filename_prefix",
                "group_verified": bool(case),
            }
        )
    if not samples:
        raise RuntimeError(f"No BUSBRA classification images found under {root}")
    return samples


def _read_uclm_label_map(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    raw = csv_path.read_text(encoding="utf-8", errors="ignore")
    delimiter = ";" if raw.count(";") >= raw.count(",") else ","
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            normalized = _normalise_csv_row(row)
            image_key = next(
                (
                    key
                    for key in normalized
                    if key
                    in {
                        "image",
                        "filename",
                        "file",
                        "name",
                        "img",
                        "imagefile",
                        "image_name",
                    }
                ),
                None,
            )
            label_key = next(
                (
                    key
                    for key in normalized
                    if key in {"label", "class", "pathology", "category", "diagnosis", "type"}
                ),
                None,
            )
            if image_key is None or label_key is None:
                break
            key = normalized[image_key]
            mapping[key] = normalized[label_key].lower()
            mapping[Path(key).name] = normalized[label_key].lower()
            mapping[Path(key).stem] = normalized[label_key].lower()
    return mapping


def _resolve_uclm_root(root: Path) -> Path:
    if (root / "images").is_dir():
        return root
    candidates = sorted(path.parent for path in root.rglob("INFO.csv") if (path.parent / "images").is_dir())
    if not candidates:
        raise RuntimeError(f"No BUS-UCLM images/INFO.csv layout found under {root}")
    return candidates[0]


def scan_bus_uclm(root: str | Path) -> list[Sample]:
    root = _resolve_uclm_root(Path(root))
    images_dir = root / "images"
    label_map = _read_uclm_label_map(root / "INFO.csv")
    samples: list[Sample] = []
    for image_path in sorted(images_dir.glob("*.png")):
        label_name = (
            label_map.get(image_path.name)
            or label_map.get(image_path.stem)
            or ""
        ).strip().lower()
        patient_id = image_path.stem.split("_")[0]
        if label_name == "normal":
            continue
        if label_name not in {"benign", "malignant"}:
            continue
        samples.append(
            {
                "image": str(image_path),
                "label": 0 if label_name == "benign" else 1,
                "sample_id": image_path.stem,
                "group_id": patient_id,
                "patient_id": patient_id,
                "group_source": "filename_patient_prefix",
                "group_verified": True,
            }
        )
    if not samples:
        raise RuntimeError(f"No BUS-UCLM classification images found under {root}")
    return samples


def scan_dataset(root: str | Path, dataset_name: str) -> list[Sample]:
    normalized = dataset_name.upper().replace("-", "").replace("_", "")
    if normalized == "BUSI":
        return scan_busi(root)
    if normalized == "BUSBRA":
        return scan_busbra(root)
    if normalized in {"BUSUCLM", "UCLM"}:
        return scan_bus_uclm(root)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _pixel_digest(path: Path) -> str:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        payload = (
            f"{grayscale.width}x{grayscale.height}:".encode("ascii")
            + grayscale.tobytes()
        )
    return hashlib.sha256(payload).hexdigest()


def dataset_audit(samples: Sequence[Sample]) -> dict[str, object]:
    labels = [int(sample["label"]) for sample in samples]
    groups = [str(sample.get("group_id", sample["image"])) for sample in samples]
    paths = [str(sample["image"]) for sample in samples]
    sample_ids = [
        str(sample.get("sample_id", Path(str(sample["image"])).stem)) for sample in samples
    ]
    missing = [path for path in paths if not Path(path).exists()]
    duplicate_paths = sorted({path for path, count in Counter(paths).items() if count > 1})
    duplicate_sample_ids = sorted(
        {sample_id for sample_id, count in Counter(sample_ids).items() if count > 1}
    )
    group_labels: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        group_labels[str(sample.get("group_id", sample["image"]))].add(int(sample["label"]))
    conflicting_groups = sorted(group for group, values in group_labels.items() if len(values) > 1)

    hashes: dict[str, list[int]] = defaultdict(list)
    for index, path in enumerate(paths):
        if Path(path).is_file():
            hashes[_pixel_digest(Path(path))].append(index)
    duplicate_pixel_groups = [
        {
            "sha256": digest,
            "sample_ids": [sample_ids[index] for index in indices],
            "paths": [paths[index] for index in indices],
        }
        for digest, indices in sorted(hashes.items())
        if len(indices) > 1
    ]
    group_sources = Counter(str(sample.get("group_source", "unspecified")) for sample in samples)
    unverified_groups = sorted(
        {
            str(sample.get("group_id", ""))
            for sample in samples
            if not bool(sample.get("group_verified", False))
        }
    )
    return {
        "n_images": len(samples),
        "n_benign": int(sum(label == 0 for label in labels)),
        "n_malignant": int(sum(label == 1 for label in labels)),
        "n_groups": len(set(groups)),
        "missing_files": missing,
        "duplicate_paths": duplicate_paths,
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_pixel_groups": duplicate_pixel_groups,
        "groups_with_conflicting_labels": conflicting_groups,
        "group_sources": dict(sorted(group_sources.items())),
        "unverified_group_ids": unverified_groups,
    }


def make_folds(
    samples: Sequence[Sample],
    k: int,
    seed: int,
    strategy: str = "stratified_image",
) -> list[tuple[list[Sample], list[Sample]]]:
    strategy = strategy.lower()
    if strategy not in {"stratified_image", "stratified_group"}:
        raise ValueError("split strategy must be stratified_image or stratified_group")
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=int)
    class_counts = np.bincount(labels, minlength=2)
    if int(class_counts.min()) < k:
        raise ValueError(
            f"Each class must contain at least k={k} images; observed counts={class_counts.tolist()}"
        )
    indices = np.arange(len(samples))
    if strategy == "stratified_image":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        split_iterator = splitter.split(indices, labels)
    else:
        groups = np.asarray([str(sample.get("group_id", "")) for sample in samples])
        group_labels: dict[str, set[int]] = defaultdict(set)
        for group, label in zip(groups, labels, strict=True):
            group_labels[str(group)].add(int(label))
        conflicting = [group for group, values in group_labels.items() if len(values) > 1]
        if conflicting:
            raise ValueError(
                "Stratified group splitting requires one label per group; conflicts: "
                + ", ".join(sorted(conflicting)[:5])
            )
        grouped_class_counts = Counter(next(iter(values)) for values in group_labels.values())
        if min(grouped_class_counts.get(0, 0), grouped_class_counts.get(1, 0)) < k:
            raise ValueError(
                f"Each class must contain at least k={k} groups; "
                f"observed group counts={dict(grouped_class_counts)}"
            )
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        split_iterator = splitter.split(indices, labels, groups)

    folds = []
    for train_indices, validation_indices in split_iterator:
        train = [dict(samples[index]) for index in train_indices]
        validation = [dict(samples[index]) for index in validation_indices]
        folds.append((train, validation))
    return folds


def validate_no_group_leakage(
    train_samples: Sequence[Sample], val_samples: Sequence[Sample]
) -> None:
    train_groups = {str(sample.get("group_id", sample["image"])) for sample in train_samples}
    val_groups = {str(sample.get("group_id", sample["image"])) for sample in val_samples}
    overlap = train_groups & val_groups
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(f"Group leakage detected across train/validation: {preview}")


def _coarse_dropout():
    """Construct the manuscript's four 24x24 holes across Albumentations APIs."""
    parameters = inspect.signature(A.CoarseDropout).parameters
    if "num_holes_range" in parameters:
        return A.CoarseDropout(
            num_holes_range=(4, 4),
            hole_height_range=(24, 24),
            hole_width_range=(24, 24),
            p=0.2,
        )
    return A.CoarseDropout(
        min_holes=4,
        max_holes=4,
        min_height=24,
        max_height=24,
        min_width=24,
        max_width=24,
        p=0.2,
    )


def _gauss_noise():
    """Use a pixel-domain variance range of [10, 40] across APIs."""
    parameters = inspect.signature(A.GaussNoise).parameters
    if "var_limit" in parameters:
        return A.GaussNoise(var_limit=(10.0, 40.0), mean=0.0, p=0.5)
    return A.GaussNoise(
        std_range=(float(np.sqrt(10.0) / 255.0), float(np.sqrt(40.0) / 255.0)),
        mean_range=(0.0, 0.0),
        p=0.5,
    )


def build_transforms(img_size: int, training: bool, strong: bool = True):
    if A is None or ToTensorV2 is None:
        return None
    operations = [A.Resize(img_size, img_size)]
    if training:
        operations.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=20,
                    border_mode=0,
                    p=0.5,
                ),
            ]
        )
        if strong:
            operations.extend(
                [
                    A.OneOf(
                        [_gauss_noise(), A.GaussianBlur(blur_limit=(3, 5), p=0.5)],
                        p=0.4,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=0.4,
                    ),
                    A.CLAHE(clip_limit=3.0, p=0.3),
                    _coarse_dropout(),
                ]
            )
    operations.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(operations)


class BreastUltrasoundDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        img_size: int,
        transform=None,
        return_metadata: bool = False,
    ) -> None:
        self.samples = [dict(sample) for sample in samples]
        self.img_size = img_size
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path = str(sample["image"])
        image = np.asarray(Image.open(image_path).convert("L"))
        image = np.stack([image, image, image], axis=-1)

        if self.transform is not None:
            tensor = self.transform(image=image)["image"].float()
        else:
            resampling = getattr(Image, "Resampling", Image)
            resized = Image.fromarray(image).resize(
                (self.img_size, self.img_size), resampling.BILINEAR
            )
            array = np.asarray(resized).astype(np.float32) / 255.0
            array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
                [0.229, 0.224, 0.225], dtype=np.float32
            )
            tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()

        label = int(sample["label"])
        if self.return_metadata:
            metadata = {
                "image": image_path,
                "sample_id": str(sample.get("sample_id", Path(image_path).stem)),
                "group_id": str(sample.get("group_id", "")),
                "group_source": str(sample.get("group_source", "")),
            }
            return tensor, label, metadata
        return tensor, label

from __future__ import annotations

import csv

from PIL import Image

import suanet.data as data


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=127).save(path)


def test_busi_scanner_excludes_masks_and_normal_class(tmp_path):
    _image(tmp_path / "benign" / "benign (1).png")
    _image(tmp_path / "benign" / "benign (1)_mask.png")
    _image(tmp_path / "malignant" / "malignant (1).png")
    _image(tmp_path / "normal" / "normal (1).png")
    samples = data.scan_busi(tmp_path)
    assert [sample["label"] for sample in samples] == [0, 1]
    assert all("mask" not in str(sample["image"]).lower() for sample in samples)


def test_busbra_birads_mapping(tmp_path):
    _image(tmp_path / "2" / "bus_001-1.png")
    _image(tmp_path / "5" / "bus_002-1.png")
    samples = data.scan_busbra(tmp_path)
    assert [sample["label"] for sample in samples] == [0, 1]
    assert [sample["group_id"] for sample in samples] == ["bus_001", "bus_002"]


def test_bus_uclm_csv_labels_and_patient_groups(tmp_path):
    _image(tmp_path / "images" / "AAAA_001.png")
    _image(tmp_path / "images" / "BBBB_001.png")
    with (tmp_path / "INFO.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Image", "Pathology"], delimiter=";")
        writer.writeheader()
        writer.writerow({"Image": "AAAA_001.png", "Pathology": "benign"})
        writer.writerow({"Image": "BBBB_001.png", "Pathology": "malignant"})
    samples = data.scan_bus_uclm(tmp_path)
    assert [sample["label"] for sample in samples] == [0, 1]
    assert [sample["patient_id"] for sample in samples] == ["AAAA", "BBBB"]
    assert [sample["group_id"] for sample in samples] == ["AAAA", "BBBB"]


def test_coarse_dropout_supports_both_albumentations_apis(monkeypatch):
    class VersionTwoDropout:
        def __init__(
            self,
            num_holes_range,
            hole_height_range,
            hole_width_range,
            p,
        ):
            self.holes = num_holes_range

    class VersionOneDropout:
        def __init__(
            self,
            max_holes,
            max_height,
            max_width,
            min_holes,
            min_height,
            min_width,
            p,
        ):
            self.holes = (min_holes, max_holes)

    class FakeAlbumentations:
        CoarseDropout = VersionTwoDropout

    monkeypatch.setattr(data, "A", FakeAlbumentations)
    assert data._coarse_dropout().holes == (4, 4)

    FakeAlbumentations.CoarseDropout = VersionOneDropout
    assert data._coarse_dropout().holes == (4, 4)

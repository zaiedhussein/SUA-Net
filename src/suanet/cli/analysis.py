from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from suanet.analysis import (
    load_prediction_directory,
    paired_mc_across_datasets,
    threshold_operating_points,
)
from suanet.config import load_yaml
from suanet.utils import save_json


def _resolve(base: Path, value: object) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run manuscript threshold and paired MC-Dropout analyses"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    payload = load_yaml(config_path)
    base = config_path.parent
    result_dirs = {
        name: _resolve(base, value) for name, value in payload["result_dirs"].items()
    }
    output_dir = Path(_resolve(base, payload.get("output_dir", "results/analysis")))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: load_prediction_directory(path) for name, path in result_dirs.items()
    }
    thresholds = {}
    for name, fold_outputs in outputs.items():
        y_true = np.concatenate([output.y_true for output in fold_outputs])
        y_prob = np.concatenate([output.y_prob for output in fold_outputs])
        thresholds[name] = threshold_operating_points(y_true, y_prob)
    save_json(thresholds, output_dir / "threshold_analysis.json")
    paired = paired_mc_across_datasets(
        outputs,
        n_bootstrap=int(payload.get("n_bootstrap", 2000)),
        seed=int(payload.get("seed", 42)),
    )
    save_json(paired, output_dir / "mc_dropout_paired_analysis.json")
    print(output_dir)


if __name__ == "__main__":
    main()

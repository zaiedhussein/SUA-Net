from __future__ import annotations

import argparse
from pathlib import Path

from suanet.config import load_yaml
from suanet.experiments import run_generalization


def _absolute(base: Path, value: object) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


def _resolve_paths(payload: dict, config_path: Path) -> dict:
    base = config_path.resolve().parent
    for key in ("base_config", "output_dir"):
        if key in payload:
            payload[key] = _absolute(base, payload[key])
    for mapping_key in ("datasets", "model_dirs", "source_configs"):
        if mapping_key in payload:
            payload[mapping_key] = {
                name: _absolute(base, value) for name, value in payload[mapping_key].items()
            }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-dataset SUA-Net generalization")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    payload = _resolve_paths(load_yaml(config_path), config_path)
    result = run_generalization(payload)
    print(result["cross_dataset"])


if __name__ == "__main__":
    main()

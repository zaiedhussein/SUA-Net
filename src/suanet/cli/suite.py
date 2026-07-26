from __future__ import annotations

import argparse
from pathlib import Path

from suanet.config import load_yaml
from suanet.experiments import run_training_suite


def _absolute(base: Path, value: object) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete SUA-Net three-dataset training suite"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    payload = load_yaml(config_path)
    payload["experiments"] = {
        name: _absolute(config_path.parent, value) for name, value in payload["experiments"].items()
    }
    if "output_dir" in payload:
        payload["output_dir"] = _absolute(config_path.parent, payload["output_dir"])
    result = run_training_suite(payload)
    print(f"Completed {len(result['datasets'])} datasets")


if __name__ == "__main__":
    main()

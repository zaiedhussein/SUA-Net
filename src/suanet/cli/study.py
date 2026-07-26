from __future__ import annotations

import argparse
from pathlib import Path

from suanet.config import load_yaml
from suanet.studies import (
    run_configured_study,
    run_mc_sample_sensitivity,
    run_multi_dataset_sensitivity,
)


def _resolve(base: Path, value: object) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a configured SUA-Net manuscript sensitivity study"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    payload = load_yaml(config_path)
    base = config_path.parent
    mode = str(payload.get("mode", "configured"))
    payload["output_dir"] = _resolve(base, payload.get("output_dir", "results/study"))
    if mode == "configured":
        payload["experiment_config"] = _resolve(base, payload["experiment_config"])
        result = run_configured_study(payload)
    elif mode == "imbalance":
        payload["experiments"] = {
            name: _resolve(base, value) for name, value in payload["experiments"].items()
        }
        result = run_multi_dataset_sensitivity(payload)
    elif mode == "mc_sample_size":
        payload["configs"] = {
            name: _resolve(base, value) for name, value in payload["configs"].items()
        }
        payload["result_dirs"] = {
            name: _resolve(base, value) for name, value in payload["result_dirs"].items()
        }
        result = run_mc_sample_sensitivity(payload)
    else:
        raise ValueError("mode must be configured, imbalance, or mc_sample_size")
    print(result.keys())


if __name__ == "__main__":
    main()

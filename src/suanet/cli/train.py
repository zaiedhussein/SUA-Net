from __future__ import annotations

import argparse

from suanet.config import load_experiment_config
from suanet.experiments import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SUA-Net with cross-validation")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    result = run_cross_validation(config)
    print(result["summary"])


if __name__ == "__main__":
    main()

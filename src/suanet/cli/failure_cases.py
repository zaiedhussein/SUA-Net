from __future__ import annotations

import argparse

from suanet.config import load_experiment_config
from suanet.experiments import generate_failure_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out Grad-CAM failure-case panels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=6)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    result = generate_failure_cases(config, args.fold, args.max_cases)
    print(result["figure"])


if __name__ == "__main__":
    main()

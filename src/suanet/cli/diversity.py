from __future__ import annotations

import argparse
from pathlib import Path

from suanet.config import load_yaml
from suanet.diversity import run_diversity_analysis


def _resolve(base: Path, value: object) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze BUS dataset diversity and representation shift"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    payload = load_yaml(config_path)
    base = config_path.parent
    payload["datasets"] = {
        name: _resolve(base, value) for name, value in payload["datasets"].items()
    }
    payload["output_dir"] = _resolve(
        base, payload.get("output_dir", "results/diversity")
    )
    result = run_diversity_analysis(payload)
    print(result["centroid_distances"])


if __name__ == "__main__":
    main()

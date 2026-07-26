from __future__ import annotations

import argparse
from pathlib import Path

from suanet.config import apply_overrides, load_experiment_config, load_yaml
from suanet.experiments import run_ablation


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SUA-Net ablation experiments")
    parser.add_argument("--config", required=True, help="Path to an ablation YAML file")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    payload = load_yaml(config_path)
    base = config_path.parent
    experiment_path = _resolve(base, str(payload["experiment_config"]))
    config = load_experiment_config(experiment_path)
    apply_overrides(config, payload.get("overrides"))
    if "output_dir" in payload:
        config.runtime.output_dir = str(_resolve(base, str(payload["output_dir"])))
    result = run_ablation(
        config,
        payload.get("variants"),
        generate_gradcam=bool(payload.get("generate_gradcam", True)),
        gradcam_samples=int(payload.get("gradcam_samples", 8)),
    )
    print(f"Completed {len(result['variants'])} variants")


if __name__ == "__main__":
    main()

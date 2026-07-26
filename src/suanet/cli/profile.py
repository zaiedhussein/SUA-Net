from __future__ import annotations

import argparse
from pathlib import Path

from suanet.config import load_experiment_config, load_yaml
from suanet.profiling import profile_architectures


def _resolve(base: Path, value: object) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else (base / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SUA-Net architecture variants")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    payload = load_yaml(config_path)
    base = config_path.parent
    experiment = load_experiment_config(_resolve(base, payload["experiment_config"]))
    result = profile_architectures(
        experiment,
        payload["variants"],
        output_dir=_resolve(base, payload.get("output_dir", "results/profiling")),
        warmup=int(payload.get("warmup", 50)),
        iterations=int(payload.get("iterations", 500)),
        repetitions=int(payload.get("repetitions", 5)),
        measure_runtime=bool(payload.get("measure_runtime", True)),
        require_cuda=bool(payload.get("require_cuda", True)),
        mc_passes=[int(value) for value in payload.get("mc_passes", [10, 20, 30, 50])],
    )
    print(f"Profiled {len(result['architectures'])} variants")


if __name__ == "__main__":
    main()

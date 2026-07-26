.PHONY: install test lint format hygiene audit train-suite analyze generalize ablate design backbones imbalance mc-sensitivity profile diversity uncertainty

install:
	python -m pip install -e ".[dev]"

test:
	pytest --cov=suanet --cov-report=term-missing

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

hygiene:
	python scripts/check_release_hygiene.py

audit:
	suanet-audit --config configs/train/busi.yaml
	suanet-audit --config configs/train/busbra.yaml
	suanet-audit --config configs/train/bus_uclm.yaml

train-suite:
	suanet-train-suite --config configs/train_suite.yaml

analyze:
	suanet-analyze --config configs/analysis.yaml

generalize:
	suanet-generalize --config configs/generalization.yaml

ablate:
	suanet-ablate --config configs/ablation/busi.yaml
	suanet-ablate --config configs/ablation/busbra.yaml

design:
	suanet-ablate --config configs/design/busbra.yaml

backbones:
	suanet-study --config configs/backbone/busi.yaml

imbalance:
	suanet-study --config configs/sensitivity/imbalance.yaml

mc-sensitivity:
	suanet-study --config configs/sensitivity/mc_sample_size.yaml

profile:
	suanet-profile --config configs/profiling.yaml

diversity:
	suanet-diversity --config configs/diversity.yaml

uncertainty:
	suanet-uncertainty-cases --config configs/train/busbra.yaml --fold 1
	suanet-uncertainty-cases --config configs/train/bus_uclm.yaml --fold 1

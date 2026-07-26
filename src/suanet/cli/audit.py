from __future__ import annotations

import argparse
import json

from suanet.config import load_experiment_config
from suanet.data import dataset_audit, make_folds, scan_dataset, validate_no_group_leakage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dataset discovery and cross-validation splits"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    samples = scan_dataset(config.dataset.root, config.dataset.name)
    audit = dataset_audit(samples)
    folds = make_folds(
        samples, config.training.k_folds, config.training.seed, config.dataset.split_strategy
    )
    fold_report = []
    for index, (train, validation) in enumerate(folds, start=1):
        if config.dataset.split_strategy == "stratified_group":
            validate_no_group_leakage(train, validation)
        fold_report.append(
            {
                "fold": index,
                "train_images": len(train),
                "validation_images": len(validation),
                "train_positive": sum(int(s["label"]) for s in train),
                "validation_positive": sum(int(s["label"]) for s in validation),
            }
        )
    print(json.dumps({"audit": audit, "folds": fold_report}, indent=2))


if __name__ == "__main__":
    main()

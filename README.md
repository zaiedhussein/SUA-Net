# SUA-Net


> *Speckle Aware Attention and Multi Scale Pooling for Robust Breast Ultrasound
> Lesion Classification Across Multiple Benchmarks*

SUA-Net classifies breast-ultrasound lesions as benign or malignant using an
ImageNet-pretrained EfficientNet-B0 encoder followed by Dual-Path Lightweight
Aggregation (DLA), Speckle Variance Attention (SVA), and a Multi-Granularity
Pooling (MGP) head. Monte Carlo Dropout is an optional uncertainty mechanism;
it is not presented as a performance-improvement method.

> Research-use notice: this retrospective research software is not a medical
> device and must not be used for autonomous clinical diagnosis.



`src/suanet/model.py` is the single architecture source of truth. Training,
ablation, generalization, profiling, uncertainty visualization, and all
sensitivity studies import it rather than carrying private model copies.

## Manuscript protocol at a glance

| Item | Public configuration |
|---|---|
| Task | Binary benign–malignant lesion classification; normal images excluded |
| Input | Grayscale replicated to 3 channels, resized to 224×224, ImageNet normalized |
| Backbone | EfficientNet-B0, pretrained, first two `blocks` frozen |
| DLA | 1×1 pointwise path; 3×3 depthwise path with dilation 3; learned gated residual fusion |
| SVA | 7×7 local variance, reduction ratio 8, joint channel–spatial attention |
| MGP | Population mean, maximum, and population standard deviation; hidden dimension 640 |
| Loss | Label-smoothed focal loss, γ=2.0, smoothing=0.05 |
| Optimization | AdamW, LR 3e-4, weight decay 1e-4, cosine warm restarts, 40 epochs |
| Batch | 16 with 2-step accumulation (effective batch 32) |
| Control | FP16 AMP, gradient norm 1.0, accuracy checkpointing, patience 20 |
| Validation | 5 folds, seed 42 |
| BUSI | Stratified image-wise folds; 647 lesions; patient identity unavailable |
| BUSBRA | Patient-wise `StratifiedGroupKFold` using the metadata `Case` field |
| BUS-UCLM | Patient-wise `StratifiedGroupKFold` using the filename patient prefix |
| MC-Dropout | T=20; population probability dispersion averaged over both classes |

The BUSBRA command refuses to run the manuscript protocol when it cannot find
verified `Case` metadata. It never silently substitutes a filename-based
image-wise split.

## Installation

Python 3.10 or newer is required. Clone the repository using the URL
shown by GitHub's **Code** button, then run:

```bash
cd SUA-Net
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Development checks:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests scripts
python -m pytest
python scripts/check_release_hygiene.py
```

On Kaggle or Colab, clone and install this repository rather than copying
the implementation into a notebook cell.

## Data

The manuscript uses the public BUSI, BUSBRA, and BUS-UCLM releases. Edit only
the dataset roots and output directories in the YAML files. The scanners
exclude masks and normal images as appropriate and retain patient/case groups.

Before training:

```bash
suanet-audit --config configs/train/busi.yaml
suanet-audit --config configs/train/busbra.yaml
suanet-audit --config configs/train/bus_uclm.yaml
```

The audit reports class and group counts, missing paths, duplicate identifiers,
exact pixel-duplicate groups, conflicting patient labels, and whether grouping
comes from verified metadata. The manuscript retained the public BUSI release
without deduplication; BUSI must therefore be interpreted as a potentially
optimistic image-wise benchmark.

Dataset download pages:

- [BUSI](https://www.kaggle.com/datasets/joydhar/breast-cancer-ultrasound-dataset)
- [BUSBRA](https://www.kaggle.com/datasets/orvile/bus-bra-a-breast-ultrasound-dataset)
- [BUS-UCLM](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset)

Dataset files are not redistributed. Users are responsible for obtaining each
dataset from its host and complying with the applicable terms.

## Reproduce the manuscript workflows

First run the three primary models:

```bash
suanet-train-suite --config configs/train_suite.yaml
```

Then run the analyses that consume their fold predictions and checkpoints:

```bash
suanet-analyze --config configs/analysis.yaml
suanet-generalize --config configs/generalization.yaml
suanet-study --config configs/sensitivity/mc_sample_size.yaml
```

Additional training studies:

```bash
# Architecture ablations on both reported datasets
suanet-ablate --config configs/ablation/busi.yaml
suanet-ablate --config configs/ablation/busbra.yaml

# Focused BUSBRA design/hyperparameter ablation
suanet-ablate --config configs/design/busbra.yaml

# BUSI backbone comparison
suanet-study --config configs/backbone/busi.yaml

# Baseline, weighted-focal, and balanced-sampler sensitivity
suanet-study --config configs/sensitivity/imbalance.yaml
```

Hardware-independent complexity and controlled deployment profiling:

```bash
suanet-profile --config configs/profiling.yaml
```

The default profiling configuration enforces CUDA, FP32, batch size 1, 50
warm-ups, five repetitions, and 500 timed forward passes per repetition.
Latency numbers depend on the exact hardware and software environment; the
command records both.

Dataset diversity and representation shift:

```bash
suanet-diversity --config configs/diversity.yaml
```

High-uncertainty held-out examples:

```bash
suanet-uncertainty-cases --config configs/train/busbra.yaml --fold 1 --max-cases 6
suanet-uncertainty-cases --config configs/train/bus_uclm.yaml --fold 1 --max-cases 6
```

These panels contain the ultrasound, Grad-CAM, SVA variance view, truth,
prediction, MC malignancy probability, and predictive dispersion. They are
qualitative and are not lesion-localization evaluations.

## Primary outputs

Every training fold stores:

```text
fold_1/
├── best_model.pth
├── split_manifest.json
├── history.json
├── metrics.json
├── predictions.csv
├── training_curves.png
├── roc_curve.png
├── pr_curve.png
└── confusion_matrix.png
```

`predictions.csv` includes sample, image, and group identifiers; deterministic
probabilities; separately named MC probabilities; and uncertainty. Dataset
outputs include the resolved configuration, data audit, mean/SD/95% t
intervals, pooled bootstrap AUC interval, threshold operating points, ROC/PR,
confusion, calibration, and fold-stability figures.

The complete suite additionally writes a 2,000-resample paired deterministic
vs MC analysis. BUSBRA and BUS-UCLM use patient/group-clustered resampling;
BUSI uses image resampling. AUC is the primary endpoint. F1, MCC, Brier,
ECE-10, and exact McNemar tests are secondary families, with Holm correction
applied across the three datasets within each family.

## Reproducibility boundary

The repository reproduces the algorithms and protocols; it does not hard-code
the numerical manuscript tables. Exact values require the same dataset
releases, folds, pretrained weights, library stack, checkpoints, and hardware.
The manuscript notes that the pretrained-weight identifier used for the
original diversity analysis was not recorded. New public runs record the
resolved timm version and pretrained configuration, but that missing historical
identifier cannot be reconstructed honestly.

The BUSI literature comparison summarizes external publications and is not a
controlled reimplementation; it therefore has no training command here.

Critical manuscript definitions are enforced by
`tests/test_manuscript_alignment.py` and `tests/test_protocols.py`; the
workflow-to-command mapping is contained in this README.

## Release

Large checkpoints belong in a versioned GitHub Release or archival repository,
not normal Git history. Before tagging the paper release, run the lint, test,
build, and hygiene commands shown above; generate a checksum manifest with
`python scripts/release_manifest.py`; and add the verified GitHub URL and
article DOI to `CITATION.cff` when those identifiers exist.

## License

The implementation is released under the MIT License. Dataset, pretrained
weight, and third-party dependency licenses remain separate.

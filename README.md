# GraphKAN

GraphKAN combines graph filtering with grouped rational activations for EEG emotion classification. This repository provides configuration-driven subject-dependent (SD), subject-independent (SI), and ablation experiments for SEED, SEED-IV, and EEG-Emotion.

## Installation

Use Python 3.11 and a CUDA-enabled PyTorch environment. Install the Python dependencies and install [rational_kat_cu](https://github.com/Adamdad/rational_kat_cu) separately, following [Installation](docs/INSTALL.md). The external package is imported as `kat_rational`. Its source code, initialization files, and compiled extensions are not distributed here.

Two model-specific ablations, grouped B-spline and ungrouped edgewise rational functions, have local implementations in `src/gkan/activations.py`. All standard rational configurations use the installed `kat_rational.KAT_Group`.

## Data preparation

**SEED and SEED-IV are not included.** Obtain them from the [official SEED project](https://bcmi.sjtu.edu.cn/home/seed/) under its access requirements. Set `dataset.root` in the selected JSON configuration to the downloaded feature directory. EEG-Emotion signals, features, and participant information are also excluded from this source repository.

Default paths are relative to this repository:

```text
data/SEED/ExtractedFeatures/label.mat
data/SEED/ExtractedFeatures/<subject>_<date>.mat
data/SEED-IV/<session>/<subject>_<date>.mat
data/EEG-Emotion/features/manifest.json
data/EEG-Emotion/features/sub<subject>_session1.npz
```

EEG-Emotion uses subject IDs **1-20** and session **1**. Its DE extraction bands are 1-4, 4-8, 8-14, 14-30, and **30-48 Hz**. Existing official SEED/SEED-IV DE features are read as supplied; the preprocessing configuration does not change their frequency content. See [Data](docs/DATA.md) and [EEG-Emotion](docs/EEG_EMOTION.md).

## Training

Run commands from this repository directory:

```powershell
python scripts/train.py --config Configs/seed_sd.json
python scripts/train.py --config Configs/seed_si.json
```

- **SD:** set `protocol.subject` and `protocol.session`. One invocation trains only that subject/session. SEED uses the first nine trials for training; SEED-IV uses the first four trials per class. EEG-Emotion uses five whole-trial folds within the selected subject's session.
- **SI:** set `protocol.test_subject`. SEED/SEED-IV train on all three sessions of the other 14 subjects and test on all three sessions of the selected subject. EEG-Emotion uses the other 19 subjects for training and the selected subject for testing.
- Defaults are **100 epochs**, batch size **128**, and one random seed, `training.seed=123`. Subject/session/seed lists are not accepted.
- All runs use **global z-score normalization over the complete dataset cohort and all sessions**. The full feature dataset is required even when running a single SD target.
- `training.selection="test_peak"` retains the highest test-accuracy checkpoint, choosing the earliest epoch in a tie. `"final"` selects the last epoch. Global normalization includes unlabeled test features; test-peak selection uses test labels. Report these choices explicitly when presenting results.

For PyCharm, select `scripts/train.py` as the script, this directory as the working directory, and `--config Configs/seed_sd.json` as the parameters.

## Ablations

Each file defaults to SEED-SD, subject 1/session 1. Edit its target fields and run one configuration at a time.

| Model | Configuration in `Configs/ablations/` |
|---|---|
| 5-Swish | `seed_sd_5_swish.json` |
| 5-B-spline | `seed_sd_5_bspline.json` |
| 5-Identity | `seed_sd_5_identity.json` |
| 1-Swish | `seed_sd_1_swish.json` |
| w/o KAN | `seed_sd_wo_kan.json` |
| w/o Dyn-Adj | `seed_sd_wo_dyn_adj.json` |
| w/o group | `seed_sd_wo_group.json` |

```powershell
python scripts/train.py --config Configs/ablations/seed_sd_5_bspline.json
```

Architecture, ablation definitions, and optimization settings are listed in [Model settings](docs/PARAMETERS.md).

## Checks and outputs

```powershell
python scripts/train.py --config Configs/seed_si.json --list-folds
python scripts/train.py --config Configs/seed_sd.json --smoke
python -m unittest discover -s tests -v
```

Smoke mode runs two epochs on a small subset of the target's first fold, while retaining the full normalization scope. Outputs under `outputs/` include configuration, data fingerprints, splits, normalization statistics, training histories, checkpoints, and predictions. Generated outputs and datasets are ignored by Git.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE.md). The code license does not cover datasets or external dependencies.

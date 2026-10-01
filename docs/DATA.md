# Data

## Obtaining SEED and SEED-IV

No SEED or SEED-IV signals, feature files, labels.mat files, or trained weights are distributed with this repository. Request access and download data from the [official SEED project](https://bcmi.sjtu.edu.cn/home/seed/). Set `dataset.root` to your local feature directory after obtaining access.

### SEED

The directory contains `label.mat` and `<subject>_<YYYYMMDD>.mat` files. Subjects are numbered 1-15. Three files per subject are sorted by date to determine sessions1-3. The loader reads `de_LDS1` through `de_LDS15`, stored as `[62,windows,5]`, and converts them to `[windows,62,5]`. Labels -1/0/+1 map to negative/neutral/positive (0/1/2).

SD trains on the first nine trials and tests on the remaining six for the selected subject/session. SI trains on all three sessions of the other 14 subjects and tests on all three sessions of the selected subject.

### SEED-IV

The feature directory has subdirectories `1`, `2`, and `3` for sessions. Each contains one MAT file per subject, with `de_LDS1` through `de_LDS24`. Labels 0/1/2/3 correspond to neutral/sad/fear/happy; each session has its own trial-label sequence.

SD takes the first four trials per class for training and the remaining two for testing. SI uses the other 14 subjects across all three sessions for training. A flattened `all/` copy is not loaded.

Official pre-extracted features are read without re-filtering or re-smoothing. Changing the EEG-Emotion extraction bands does not change the frequency content of these MAT arrays. A different band definition requires feature extraction from the corresponding raw signals. Refer to the [SEED-IV documentation](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) for the official preprocessing and window settings.

## EEG-Emotion features

Subjects are numbered **1-20**, with session **1**. Each `subN_session1.npz` contains:

- `x`: unnormalized float32 features with shape `[windows,62,5]`.
- `y`: integer class IDs, shape `[windows]`.
- `trial`: positive integer trial IDs, shape `[windows]`.

All windows in one trial have one label. Channel names/order must match `assets/eeg_emotion_montage.json`. The manifest declares `schema_version=1`, `feature="de"`, `normalized=false`, classes, channels, records, and bands `[[1,4],[4,8],[8,14],[14,30],[30,48]]`. A schema example is provided in `examples/eeg_emotion_features.example.json`; it contains no participant data.

SD uses five whole-trial folds for one subject/session. SI trains on the other 19 subjects and tests on the selected subject. Whole trials are kept intact.

## Complete normalization cohort

All runs require the complete feature cohort because global standardization covers every subject and session, even for a single SD target. It includes unlabeled test features. Dataset preprocessing is separate from normalization during training.

The feature-conversion command below exports the full normalization cohort into NPZ format without normalization:

```powershell
python scripts/preprocess.py --config Configs/seed_sd.json --output data/SEED/portable_features
```

Update `dataset.format` to `npz_manifest` and `dataset.root` to the output directory to use those files. Never commit generated features or dataset files to the code repository.

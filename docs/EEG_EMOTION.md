# EEG-Emotion preprocessing

Use subject IDs **1-20** and session **1** throughout the feature and trial manifests. The source package contains no participant recordings, ratings, generated features, or subject-identity mapping.

## Required inputs

The EEGLAB adapter accepts continuous SET/FDT signals with 62 channels and explicit microvolt units. The channel order must match `assets/eeg_emotion_montage.json`. Artifact cleaning must already be completed and documented.

Copy `examples/eeg_emotion_trials.example.json` to a local data directory and complete it with the real trial records for all 20 subjects. The example is intentionally unconfirmed and contains no assumed labels. Each retained trial needs a subject, session, trial ID, relative SET path, start/end event codes, integer label, and boolean retention decision. Confirm the manifest only after verifying these fields and the artifact-processing status.

The supported event convention uses S1-S16 for baseline starts, S17-S32 for video starts, and S33-S48 for video ends. The adapter extracts `[start,end)` video intervals and rejects intervals crossing an EEGLAB boundary. Baseline intervals are not subtracted from DE features. Labels and retention decisions come from the explicit manifest; they are never inferred from file order or model predictions.

## Extraction

Configure `Configs/preprocess_eeglab.json` with the raw-data root, confirmed trial manifest, and a new output directory, then run:

```powershell
python scripts/preprocess_eeglab.py --config Configs/preprocess_eeglab.json
```

The output directory must not already exist. The pipeline converts microvolts to volts, resamples to128Hz if needed, filters each trial separately using fourth-order Butterworth bandpass filters with forward/backward SOS filtering, and uses one-second non-overlapping windows. The bands are:

| Band | Frequency range (Hz) |
|---|---:|
| Delta | 1-4 |
| Theta | 4-8 |
| Alpha | 8-14 |
| Beta | 14-30 |
| Gamma | **30-48** |

DE is `0.5 * ln(2*pi*e*variance)`, with population variance (`ddof=0`) and a variance floor of `1e-20`. Incomplete trailing windows are discarded. No LDS smoothing is applied.

The generated feature manifest records the frequency bands. The loader checks that these match the training configuration. Features extracted with a different upper gamma cutoff must be re-extracted; changing JSON metadata alone is not sufficient.

Raw inputs and generated `preprocessing.json` provenance are local data artifacts and are excluded from the Git repository.

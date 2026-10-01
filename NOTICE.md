# Attribution and distribution

- The GraphKAN source in this repository is licensed under the MIT License in `LICENSE`.
- Standard grouped rational activations use the external [Adamdad/rational_kat_cu](https://github.com/Adamdad/rational_kat_cu) project, imported as `kat_rational`. Users must install it separately. Its source code, initialization JSON files, and binaries are not bundled or relicensed here.
- `src/gkan/activations.py` contains two explicit ablation components: band-shared cubic B-spline activations and edgewise ungrouped rational functions. The B-spline formulation follows the Cox-de Boor recurrence and a SiLU base; see [efficient-kan](https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py) for a reference formulation. The complete external KANLinear implementation is not vendored.
- SEED and SEED-IV must be obtained from their data providers under the applicable dataset terms. The code license does not grant access to or redistribution rights for these datasets.
- EEG-Emotion recordings, participant information, features, and trained checkpoints are not included. Example manifests describe file schemas only.
- Electrode assets describe channel order and geometry, not participant signals. SEED-IV trial-label constants follow the public dataset convention, cross-checked against the [TorchEEG loader](https://torcheeg.readthedocs.io/en/v1.1.3/_modules/torcheeg/datasets/module/emotion_recognition/seed_iv.html).

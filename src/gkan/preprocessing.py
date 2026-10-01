"""Feature interchange and DE extraction from already artifact-cleaned trials."""
import argparse
import hashlib
import json
from math import gcd
from pathlib import Path
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from .config import ROOT, path, read_config, write_json
from .data import load_records


def prepare_main():
    parser = argparse.ArgumentParser(
        description="Convert validated MAT/NPZ features to portable NPZ without normalization"
    )
    parser.add_argument("--config", default=str(ROOT / "Configs/seed_sd.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    c = read_config(args.config)
    records = load_records(c, normalization_pool=True)
    output = path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    montage = json.loads(path(c["model"]["montage"]).read_text())
    items = []
    for r in records:
        name = f"sub{r.subject}_session{r.session}.npz"
        np.savez_compressed(output / name, x=r.x, y=r.y, trial=r.trial)
        items.append(
            dict(
                subject=r.subject,
                session=r.session,
                file=name,
                source=r.source,
                source_sha256=r.sha256,
            )
        )
    write_json(
        output / "manifest.json",
        dict(
            schema_version=1,
            feature=c["dataset"]["feature"],
            classes=c["dataset"]["classes"],
            channels=montage["channels"],
            normalized=False,
            **({"bands": c["dataset"]["bands"]} if c["dataset"]["name"] == "EEG-Emotion" else {}),
            records=items,
        ),
    )
    print(
        f"Wrote {len(items)} records to {output}. Set dataset.format=npz_manifest and dataset.root to this directory."
    )


def extract_de(
    signal,
    sample_rate,
    bands,
    target_rate=200,
    window_seconds=1.0,
    step_seconds=1.0,
    filter_order=4,
):
    """Natural-log Gaussian DE, population variance, explicit nonoverlap by default.

    Input is artifact-cleaned channels x time in volts. Filters never span trials.
    This is not the official LDS smoother and does not reproduce de_LDS arrays.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or sample_rate <= 0:
        raise ValueError("Invalid raw signal")
    if type(sample_rate) is not int or type(target_rate) is not int:
        raise ValueError("Sampling rates must be integers")
    if sample_rate != target_rate:
        factor = gcd(sample_rate, target_rate)
        x = resample_poly(x, target_rate // factor, sample_rate // factor, axis=-1)
    window = round(window_seconds * target_rate)
    step = round(step_seconds * target_rate)
    if window < 2 or step < 1 or x.shape[1] < window:
        raise ValueError("Invalid/insufficient window duration")
    features = []
    for low, high in bands:
        if not 0 < low < high < target_rate / 2:
            raise ValueError("Band must lie below Nyquist")
        filtered = sosfiltfilt(
            butter(
                filter_order,
                [low, high],
                btype="bandpass",
                fs=target_rate,
                output="sos",
            ),
            x,
            axis=-1,
        )
        windows = np.stack(
            [
                filtered[:, start : start + window]
                for start in range(0, x.shape[1] - window + 1, step)
            ]
        )
        variance = windows.var(axis=-1, ddof=0)
        features.append(0.5 * np.log(2 * np.pi * np.e * np.maximum(variance, 1e-20)))
    return np.stack(features, axis=-1).astype(np.float32)



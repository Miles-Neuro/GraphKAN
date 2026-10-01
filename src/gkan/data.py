"""Validated MAT-feature and portable NPZ interfaces. No dataset is bundled."""
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .config import path, cohort, sessions as all_sessions

# Trial labels from the SEED-IV release, cross-checked against TorchEEG's loader.
# 0 neutral, 1 sad, 2 fear, 3 happy. See docs/DATA.md for source URLs.
SEED_IV_LABELS = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
]


@dataclass
class Record:
    subject: int
    session: int
    x: np.ndarray
    y: np.ndarray
    trial: np.ndarray
    source: str
    sha256: str


def check_record(record, channels, features, classes):
    x, y, trial = record.x, record.y, record.trial
    if (
        x.ndim != 3
        or x.shape[1:] != (channels, features)
        or len(x) < 2
        or not np.isfinite(x).all()
    ):
        raise ValueError(
            f"{record.source}: expected finite [windows,{channels},{features}]"
        )
    if (
        y.shape != (len(x),)
        or trial.shape != y.shape
        or not np.issubdtype(y.dtype, np.integer)
        or not np.issubdtype(trial.dtype, np.integer)
    ):
        raise ValueError("Labels/trials must be aligned one-dimensional integer arrays")
    if np.any(y < 0) or np.any(y >= classes) or np.any(trial < 1):
        raise ValueError("Invalid label or trial IDs")
    for t in np.unique(trial):
        if len(np.unique(y[trial == t])) != 1:
            raise ValueError("Each trial must have one retained emotion label")


def load_mat_features(filename, labels, feature, channels):
    keys = [f"{feature}{i}" for i in range(1, len(labels) + 1)]
    mat = loadmat(filename, variable_names=keys)
    xs = []
    ys = []
    ts = []
    for i, key in enumerate(keys, 1):
        if key not in mat:
            raise KeyError(f"{filename}: missing {key}")
        x = np.asarray(mat[key], dtype=np.float32)
        if x.ndim != 3 or x.shape[0] != channels or x.shape[2] != 5:
            raise ValueError(
                f"{filename}/{key}: expected channels x windows x 5, got {x.shape}"
            )
        xs.append(x.transpose(1, 0, 2))
        ys.append(np.full(x.shape[1], labels[i - 1], np.int64))
        ts.append(np.full(x.shape[1], i, np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)


def load_records(config, normalization_pool=False):
    d = config["dataset"]
    root = path(d["root"])
    montage = json.loads(path(config["model"]["montage"]).read_text())
    channels = montage["channels"]
    p = config["protocol"]
    subjects = cohort(config) if normalization_pool or p["type"] == "SI" else [p["subject"]]
    sessions = all_sessions(config) if normalization_pool or p["type"] == "SI" else [p["session"]]
    found = {}
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root}. Set dataset.root in the JSON."
        )
    if d["format"] == "npz_manifest":
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("normalized") is not False
        ):
            raise ValueError(
                "Manifest requires schema_version=1 and normalized=false; supply unnormalized features"
            )
        if (
            manifest["channels"] != channels
            or manifest["classes"] != d["classes"]
            or manifest["feature"] != d["feature"]
        ):
            raise ValueError(
                "Manifest channel order/classes/feature must exactly match configuration"
            )
        if d['name'] == 'EEG-Emotion' and manifest.get('bands') != d['bands']:
            raise ValueError('EEG-Emotion feature bands do not match the configuration. Re-extract features with gamma 30-48 Hz; do not relabel existing arrays.')
        for item in manifest["records"]:
            key = (item["subject"], item["session"])
            if key[0] not in subjects or key[1] not in sessions:
                continue
            if key in found:
                raise ValueError("Duplicate subject/session in manifest")
            filename = (root / item["file"]).resolve()
            if not filename.is_relative_to(root.resolve()):
                raise ValueError("Manifest file must stay within dataset directory")
            with np.load(filename, allow_pickle=False) as data:
                x = np.asarray(data["x"], dtype=np.float32)
                y = data["y"].copy()
                trial = data["trial"].copy()
            found[key] = Record(
                *key,
                x,
                y,
                trial,
                str(filename),
                hashlib.sha256(filename.read_bytes()).hexdigest(),
            )
    elif d["name"] == "SEED":
        labels = np.asarray(loadmat(root / "label.mat")["label"]).reshape(-1)
        if len(labels) != 15 or set(labels.tolist()) != {-1, 0, 1}:
            raise ValueError("Invalid SEED label.mat")
        for subject in subjects:
            files = sorted(
                [
                    f
                    for f in root.glob(f"{subject}_*.mat")
                    if re.fullmatch(r"\d+_\d{8}\.mat", f.name)
                ],
                key=lambda f: f.stem.split("_")[1],
            )
            if len(files) != 3:
                raise ValueError(f"SEED subject {subject}: expected three dated files")
            for session in sessions:
                if session not in (1, 2, 3):
                    raise ValueError("SEED sessions must be 1..3")
                f = files[session - 1]
                values = load_mat_features(
                    f, (labels + 1).astype(int), d["feature"], len(channels)
                )
                found[subject, session] = Record(
                    subject,
                    session,
                    *values,
                    str(f.resolve()),
                    hashlib.sha256(f.read_bytes()).hexdigest(),
                )
    elif d["name"] == "SEED-IV":
        for session in sessions:
            if session not in (1, 2, 3):
                raise ValueError("SEED-IV sessions must be 1..3")
            for subject in subjects:
                files = list((root / str(session)).glob(f"{subject}_*.mat"))
                if len(files) != 1:
                    raise ValueError(
                        f"SEED-IV subject {subject} session {session}: expected one MAT"
                    )
                f = files[0]
                values = load_mat_features(
                    f, SEED_IV_LABELS[session - 1], d["feature"], len(channels)
                )
                found[subject, session] = Record(
                    subject,
                    session,
                    *values,
                    str(f.resolve()),
                    hashlib.sha256(f.read_bytes()).hexdigest(),
                )
    else:
        raise ValueError("EEG-Emotion requires the documented NPZ manifest adapter")
    if set(found) != {(s, t) for s in subjects for t in sessions}:
        raise ValueError("Missing configured subjects/sessions")
    records = [found[k] for k in sorted(found)]
    for record in records:
        check_record(
            record, len(channels), config["model"]["in_features"], len(d["classes"])
        )
    return records


def distance_adjacency(config):
    m = config["model"]
    montage = json.loads(path(m["montage"]).read_text())
    distance = np.asarray(montage["distance"], dtype=np.float64)
    if (
        distance.shape != (len(montage["channels"]),) * 2
        or not np.isfinite(distance).all()
        or np.any(distance < 0)
    ):
        raise ValueError("Invalid montage distance matrix")
    if not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0):
        raise ValueError("Montage distances must be symmetric with zero diagonal")
    adjacency = np.minimum(1.0, m["delta"] / (distance + np.eye(len(distance))) ** 2)
    return np.where(adjacency < m["adjacency_threshold"], 0.0, adjacency)


def pooled_statistics(chunks):
    count = 0
    mean = None
    m2 = None
    for chunk in chunks:
        x = np.asarray(chunk, dtype=np.float64)
        n = len(x)
        if not n or not np.isfinite(x).all():
            raise ValueError("Empty/nonfinite normalization data")
        local = x.mean(0, keepdims=True)
        variance = ((x - local) ** 2).sum(0, keepdims=True)
        if not count:
            count, mean, m2 = n, local, variance
            continue
        difference = local - mean
        total = count + n
        m2 += variance + difference**2 * (count * n / total)
        mean += difference * (n / total)
        count = total
    if not count:
        raise ValueError("No normalization data")
    scale = np.sqrt(np.maximum(m2 / count, 0.0))
    scale = np.where(scale < 1e-6, 1.0, scale)
    return dict(mean=mean.astype(np.float32), scale=scale.astype(np.float32)), count

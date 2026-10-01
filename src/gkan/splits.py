"""Explicit subject/session/trial-disjoint split plans; no window-level CV."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Fold:
    name: str
    subject: int
    session: int | None
    fold: int
    train: np.ndarray
    test: np.ndarray


def concatenate(records):
    data = {k: [] for k in ("x", "y", "subject", "session", "trial", "group")}
    group_offset = 0
    for r in records:
        groups = np.empty(len(r.y), dtype=np.int64)
        for trial in np.unique(r.trial):
            groups[r.trial == trial] = group_offset
            group_offset += 1
        for key, value in [
            ("x", r.x),
            ("y", r.y),
            ("subject", np.full(len(r.y), r.subject)),
            ("session", np.full(len(r.y), r.session)),
            ("trial", r.trial),
            ("group", groups),
        ]:
            data[key].append(value)
    return {k: np.concatenate(v) for k, v in data.items()}


def make_folds(data, config):
    p = config["protocol"]
    d = config["dataset"]
    result = []
    subject = p['subject'] if p['type']=='SD' else p['test_subject']
    from .config import sessions as all_sessions
    sessions = [p['session']] if p['type']=='SD' else all_sessions(config)
    available = np.isin(data["session"], sessions)
    own = available & (data["subject"] == subject)
    session = sessions[0] if len(sessions) == 1 else None
    prefix = (
        f'sub{subject}_session{session if session is not None else "pooled"}'
    )
    if p["type"] == "SI":
        splits = [(available & ~own, own)]
    elif p["sd_split"] == "first_trials":
        train = own & (data["trial"] <= p["train_trials"])
        splits = [(train, own & ~train)]
    elif p["sd_split"] == "first_per_class":
        train = np.zeros(len(own), dtype=bool)
        for s in sessions:
            for label in range(len(d["classes"])):
                eligible = own & (data["session"] == s) & (data["y"] == label)
                trials = np.unique(data["trial"][eligible])
                if len(trials) <= p["train_trials_per_class"]:
                    raise ValueError("Need held-out trials in every class")
                train |= eligible & np.isin(
                    data["trial"], trials[: p["train_trials_per_class"]]
                )
        splits = [(train, own & ~train)]
    else:
        # Stratify indivisible whole trials. 16 trials cannot be split into exact 80/20 in every fold.
        n = p["folds"]
        buckets = [[] for _ in range(n)]
        rng = np.random.default_rng(p["split_seed"])
        cursor = 0
        if n < 2 or len(np.unique(data["group"][own])) < n:
            raise ValueError("Too few trials for requested folds")
        for label in range(len(d["classes"])):
            groups = np.unique(data["group"][own & (data["y"] == label)])
            rng.shuffle(groups)
            for group in groups:
                buckets[cursor % n].append(group)
                cursor += 1
        splits = []
        for bucket in buckets:
            test = own & np.isin(data["group"], bucket)
            splits.append((own & ~test, test))
    for i, (train, test) in enumerate(splits, 1):
        if not train.any() or not test.any() or np.any(train & test):
            raise ValueError("Invalid empty/overlapping split")
        if set(data["group"][train]) & set(data["group"][test]):
            raise ValueError("Trial leakage")
        if p["type"] == "SI" and set(data["subject"][train]) & set(
            data["subject"][test]
        ):
            raise ValueError("Subject leakage")
        if set(data["y"][train]) != set(range(len(d["classes"]))):
            raise ValueError("Training split lacks a class")
        result.append(
            Fold(
                f"{prefix}_fold{i}",
                subject,
                session,
                i,
                np.flatnonzero(train),
                np.flatnonzero(test),
            )
        )
    return result

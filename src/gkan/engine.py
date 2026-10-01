"""Configuration-driven SD/SI training and auditable checkpoint selection."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .config import ROOT, path, read_config, write_json, cohort, sessions
from .data import load_records, distance_adjacency, pooled_statistics
from .model import GraphKAN
from .splits import concatenate, make_folds


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model, c):
    o = c["optimizer"]
    groups = {}
    names = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        rational = "weight_numerator" in name or "weight_denominator" in name or ".kan." in name
        decay = (
            o["rational_weight_decay"]
            if rational
            else (
                o["head_weight_decay"] if name.startswith("out.") else o["weight_decay"]
            )
        )
        if o["weights_only"] and p.ndim < 2:
            decay = 0.0
        groups.setdefault(decay, []).append(p)
        names.setdefault(decay, []).append(name)
    optimizer = torch.optim.Adam(
        [dict(params=params, weight_decay=decay) for decay, params in groups.items()],
        lr=o["lr"],
        betas=tuple(o["betas"]),
        eps=o["eps"],
        amsgrad=o["amsgrad"],
    )
    return optimizer, [
        dict(weight_decay=decay, lr=o["lr"], parameters=names[decay])
        for decay in groups
    ]


@torch.no_grad()
def evaluate(model, x, y, batch, device):
    model.eval()
    loss = 0.0
    predictions = []
    for start in range(0, len(y), batch):
        logits = model(x[start : start + batch].to(device))
        labels = y[start : start + batch].to(device)
        loss += torch.nn.functional.cross_entropy(
            logits, labels, reduction="sum"
        ).item()
        predictions.extend(logits.argmax(-1).cpu().tolist())
    accuracy = float(np.mean(np.asarray(predictions) == y.numpy()))
    return dict(loss=loss / len(y), accuracy=accuracy, predictions=predictions)


def train_epoch(model, loader, optimizer, device, smoothing):
    model.train()
    correct = 0
    loss_sum = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
        optimizer.step()
        count += len(y)
        correct += (logits.argmax(-1) == y).sum().item()
        loss_sum += loss.item() * len(y)
    return dict(train_loss=loss_sum / count, train_accuracy=correct / count)


def limited(indices, labels, maximum):
    # Smoke-only deterministic balanced subsampling; never used for full runs.
    classes = np.unique(labels[indices])
    parts = []
    for label in classes:
        parts.extend(
            indices[labels[indices] == label][: max(2, maximum // len(classes))]
        )
    return np.sort(parts)


def train_fold(data, fold, c, pooled, output, seed, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    t = c["training"]
    device = t["device"]
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "The verified kat_rational backend requires CUDA; no silent fallback is used."
        )
    train, test = fold.train, fold.test
    # Fit train-only normalization on the FULL training split, before smoke truncation.
    scope = c['normalization']['scope']
    if scope != 'global' or pooled is None:
        raise ValueError('Global standardization is mandatory')
    stats = pooled
    split = dict(
        subject=fold.subject,
        session=fold.session,
        fold=fold.fold,
        full_samples=dict(train=len(train), test=len(test)),
        train_groups=np.unique(data["group"][train]).tolist(),
        test_groups=np.unique(data["group"][test]).tolist(),
        train_subjects=np.unique(data["subject"][train]).tolist(),
        test_subjects=np.unique(data["subject"][test]).tolist(),
        train_indices=train.tolist(),
        test_indices=test.tolist(),
    )
    if smoke:
        train = limited(train, data["y"], 192)
        test = limited(test, data["y"], 192)
    split["used_samples"] = dict(train=len(train), test=len(test))
    write_json(output / "split.json", split)
    mean, scale = stats["mean"], stats["scale"]
    np.savez(output / "normalization.npz", **stats)
    x = torch.from_numpy(
        np.ascontiguousarray((data["x"][train] - mean) / scale, dtype=np.float32)
    )
    y = torch.from_numpy(data["y"][train].astype(np.int64))
    xt = torch.from_numpy(
        np.ascontiguousarray((data["x"][test] - mean) / scale, dtype=np.float32)
    )
    yt = torch.from_numpy(data["y"][test].astype(np.int64))
    seed_all(seed)
    adjacency = distance_adjacency(c)
    np.save(output / "adjacency_initial.npy", adjacency)
    model = GraphKAN(adjacency, c["model"], len(c["dataset"]["classes"])).to(device)
    optimizer, groups = make_optimizer(model, c)
    write_json(output / "optimizer_groups.json", groups)
    batch = min(t["batch_size"], 32) if smoke else t["batch_size"]
    epochs = min(t["epochs"], 2) if smoke else t["epochs"]
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=len(y) % batch == 1,
    )
    history = []
    best = -1.0
    selected_epoch = None
    selected_predictions = None
    for epoch in range(1, epochs + 1):
        row = dict(
            epoch=epoch,
            **train_epoch(model, loader, optimizer, device, t["label_smoothing"]),
        )
        if t["selection"] == "test_peak" or epoch == epochs:
            metrics = evaluate(model, xt, yt, batch, device)
            row.update(test_accuracy=metrics["accuracy"], test_loss=metrics["loss"])
            if metrics["accuracy"] > best or t["selection"] == "final":
                best = metrics["accuracy"]
                selected_epoch = epoch
                selected_predictions = metrics["predictions"]
                torch.save(
                    dict(
                        model=model.state_dict(),
                        config=c,
                        seed=seed,
                        epoch=epoch,
                        normalization={
                            k: torch.from_numpy(v) for k, v in stats.items()
                        },
                        test_used_for_selection=t["selection"] == "test_peak",
                    ),
                    output / "best.pt",
                )
        history.append(row)
        write_json(output / "history.json", history)
    torch.save(
        dict(model=model.state_dict(), config=c, seed=seed, epoch=epochs),
        output / "last.pt",
    )
    model.load_state_dict(
        torch.load(output / "best.pt", map_location=device, weights_only=True)["model"]
    )
    reload = evaluate(model, xt, yt, batch, device)
    if reload["accuracy"] != best or reload["predictions"] != selected_predictions:
        raise AssertionError("Checkpoint reload mismatch")
    trained = evaluate(model, x, y, batch, device)
    write_json(
        output / "predictions.json",
        dict(
            labels=yt.tolist(),
            predictions=reload["predictions"],
            source_indices=test.tolist(),
            subject=data["subject"][test].tolist(),
            session=data["session"][test].tolist(),
            trial=data["trial"][test].tolist(),
        ),
    )
    result = dict(
        status="passed",
        smoke=smoke,
        subject=fold.subject,
        session=fold.session,
        fold=fold.fold,
        seed=seed,
        accuracy=best,
        last_accuracy=history[-1]["test_accuracy"],
        train_accuracy=trained["accuracy"],
        selected_epoch=selected_epoch,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        stored_parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_reload="passed",
        test_used_for_selection=t["selection"] == "test_peak",
        normalization_uses_test_features=scope == "global",
        elapsed_seconds=time.perf_counter() - start,
        full_samples=split["full_samples"],
        used_samples=split["used_samples"],
    )
    write_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def aggregate(rows):
    per_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        rr = [r for r in rows if r["seed"] == seed]
        # Equal session weighting, then equal subject weighting. Within-session CV folds equal-weighted.
        subjects = {}
        for s in sorted({r["subject"] for r in rr}):
            sr = [r for r in rr if r["subject"] == s]
            session_means = [
                np.mean([r["accuracy"] for r in sr if r["session"] == sess])
                for sess in {r["session"] for r in sr}
            ]
            subjects[str(s)] = float(np.mean(session_means))
        per_seed[str(seed)] = dict(
            subject_means=subjects,
            mean_accuracy=float(np.mean(list(subjects.values()))),
            subject_sd=float(np.std(list(subjects.values()), ddof=1))
            if len(subjects) > 1
            else None,
            mean_last_accuracy=float(np.mean([r["last_accuracy"] for r in rr])),
        )
    return per_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "Configs/seed_sd.json"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="First outer fold, two epochs, <=192 windows per split; NOT an accuracy experiment",
    )
    parser.add_argument("--output")
    parser.add_argument("--fold", help="Run one exact fold name from --list-folds")
    parser.add_argument("--list-folds", action="store_true")
    args = parser.parse_args()
    c = read_config(args.config)
    torch.set_num_threads(c["training"]["num_threads"])
    records = load_records(c)
    data = concatenate(records)
    folds = make_folds(data, c)
    if args.list_folds:
        for f in folds:
            print(f.name, len(f.train), len(f.test))
        return
    if args.fold:
        folds = [f for f in folds if f.name == args.fold]
        if not folds:
            parser.error("Unknown fold name")
    if len({f.subject for f in folds}) != 1:
        raise ValueError("Only one target subject is permitted per invocation")
    if args.smoke:
        folds = folds[:1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (
        path(args.output)
        if args.output
        else path(c["output"])
        / f"{c['dataset']['name']}_{c['protocol']['type']}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", c)
    import kat_rational

    dependency = Path(inspect.getfile(kat_rational.KAT_Group))
    write_json(
        output / "environment.json",
        dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            kat_module_sha256=hashlib.sha256(dependency.read_bytes()).hexdigest(),
            montage_sha256=hashlib.sha256(
                path(c["model"]["montage"]).read_bytes()
            ).hexdigest(),
            kat_source_sha256={
                str(p.relative_to(dependency.parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in dependency.parent.rglob("*")
                if p.suffix in (".py", ".json")
            },
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "src/gkan").glob("*.py")
            },
        ),
    )
    write_json(
        output / "sources.json",
        [
            dict(
                subject=r.subject,
                session=r.session,
                file=r.source,
                sha256=r.sha256,
                samples=len(r.y),
            )
            for r in records
        ],
    )
    if c["dataset"]["format"] == "npz_manifest":
        metadata = {}
        for name in ("manifest.json", "preprocessing.json"):
            source = path(c["dataset"]["root"]) / name
            if source.exists():
                metadata[name] = json.loads(source.read_text(encoding="utf-8"))
        write_json(output / "dataset_metadata.json", metadata)
    pooled = None
    if c["normalization"]["scope"] == "global":
        pool_records = load_records(c, normalization_pool=True)
        pooled, count = pooled_statistics(r.x for r in pool_records)
        write_json(output / "normalization_sources.json", [dict(subject=r.subject, session=r.session, file=r.source, sha256=r.sha256) for r in pool_records])
        np.savez(output / "global_normalization.npz", **pooled)
        write_json(
            output / "normalization_scope.json",
            dict(
                subjects=cohort(c),
                sessions=sessions(c),
                samples=count,
                ddof=0,
                uses_test_features=True,
                uses_labels=False,
            ),
        )
    rows = []
    try:
        seed = c["training"]["seed"]
        for fold in folds:
            print(f"Running seed={seed}, {fold.name}", flush=True)
            folder = output / f"seed{seed}" / fold.name
            rows.append(train_fold(data, fold, c, pooled, folder, seed, args.smoke))
            write_json(
                output / "summary.json",
                dict(
                    status="in_progress",
                    smoke=args.smoke,
                    rows=rows,
                    per_seed=aggregate(rows),
                ),
            )
        write_json(
            output / "summary.json",
            dict(
                status="complete",
                smoke=args.smoke,
                rows=rows,
                per_seed=aggregate(rows),
                test_used_for_selection=c["training"]["selection"] == "test_peak",
                normalization_uses_test_features=c["normalization"]["scope"]
                == "global",
            ),
        )
    except Exception as exc:
        write_json(
            output / "failure.json", dict(error=type(exc).__name__, message=str(exc))
        )
        raise
    print("Results: " + str(output), flush=True)

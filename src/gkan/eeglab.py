"""Read continuous EEGLAB SET/FDT and extract explicitly labeled trials.

FDT layout follows EEGLAB eeg_getdatact: little-endian float32, channels first
in MATLAB (Fortran) order. No event-to-emotion mapping is inferred here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat
from .config import ROOT, path, write_json
from .preprocessing import extract_de


def read_set(filename):
    filename = Path(filename)
    values = loadmat(filename, simplify_cells=True)
    eeg = values.get("EEG", values)
    if int(eeg["trials"]) != 1:
        raise ValueError("This adapter supports continuous SET/FDT only")
    channels = [str(item["labels"]) for item in eeg["chanlocs"]]
    if len(channels) != int(eeg["nbchan"]) or len(set(channels)) != len(channels):
        raise ValueError("Invalid/duplicate channel labels")
    reference = eeg["data"]
    if not isinstance(reference, str) or Path(reference).name != reference:
        raise ValueError("Expected an adjacent FDT filename in EEG.data")
    fdt = filename.parent / reference
    if not fdt.exists():
        # EEGLAB files may have been renamed without updating EEG.data.
        # The audit records both identities; this does not choose a rating row.
        fdt = filename.with_suffix(".fdt")
    expected = int(eeg["nbchan"]) * int(eeg["pnts"]) * 4
    if fdt.stat().st_size != expected:
        raise ValueError("FDT byte length disagrees with SET dimensions")
    data = np.memmap(
        fdt,
        dtype="<f4",
        mode="r",
        shape=(int(eeg["nbchan"]), int(eeg["pnts"])),
        order="F",
    )
    return eeg, channels, data, fdt


def event_key(value):
    match = re.fullmatch(r"S\s*(\d+)", str(value).strip())
    return f"S{int(match[1])}" if match else str(value).strip()


def interval(eeg, start_code, end_code):
    events = eeg["event"]
    if isinstance(events, dict):
        events = [events]
    starts = [
        float(e["latency"])
        for e in events
        if event_key(e["type"]) == event_key(start_code)
    ]
    ends = [
        float(e["latency"])
        for e in events
        if event_key(e["type"]) == event_key(end_code)
    ]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"Events must occur once: {start_code}, {end_code}")
    # EEGLAB latency is 1-based. Round to nearest sample, then [start,end).
    start = int(np.floor(starts[0] - 0.5))
    end = int(np.floor(ends[0] - 0.5))
    if not 0 <= start < end <= int(eeg["pnts"]):
        raise ValueError("Invalid event interval")
    if any(
        event_key(e["type"]) == "boundary" and start < float(e["latency"]) - 1 < end
        for e in events
    ):
        raise ValueError("Trial crosses an EEGLAB discontinuity")
    return start, end


def extract(config):
    manifest = path(config["trial_manifest"])
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if m.get("confirmed") is not True or m.get("artifact_cleaned") is not True:
        raise ValueError(
            "Confirm event meanings, subject/trial selection, labels and artifact processing in trial_manifest first"
        )
    if m.get("signal_unit") != "microvolts":
        raise ValueError("SET/FDT adapter requires explicit microvolt units")
    source = path(config["root"])
    output = path(config["output"])
    montage = json.loads(path(config["montage"]).read_text(encoding="utf-8"))
    selected = m["subjects"]
    if selected != list(range(1,21)):
        raise ValueError("EEG-Emotion release cohort must be numbered 1..20")
    groups = {}
    seen = set()
    provenance = []
    for row in m["trials"]:
        key = tuple(row[k] for k in ["subject", "session", "trial"])
        if key in seen or any(type(k) is not int or k < 1 for k in key):
            raise ValueError("Duplicate/invalid trial identity")
        seen.add(key)
        if row["subject"] not in selected:
            raise ValueError("Trial outside selected cohort")
        if type(row["retain"]) is not bool:
            raise ValueError("Each trial requires an explicit retain decision")
        if not row["retain"]:
            continue
        if type(row["label"]) is not int or not 0 <= row["label"] < len(m["classes"]):
            raise ValueError("Each retained trial requires an explicit class ID")
        filename = (source / row["file"]).resolve()
        if not filename.is_relative_to(source.resolve()):
            raise ValueError("SET must stay inside dataset root")
        groups.setdefault((key[0], key[1], filename), []).append(row)
    if {k[0] for k in groups} != set(selected):
        raise ValueError("Every selected subject needs retained trials")
    output.mkdir(parents=True, exist_ok=False)
    items = []
    record_keys = set()
    for (subject, session, filename), rows in sorted(groups.items()):
        if (subject, session) in record_keys:
            raise ValueError("Only one SET per subject/session supported")
        record_keys.add((subject, session))
        eeg, channels, data, fdt = read_set(filename)
        if channels != montage["channels"]:
            raise ValueError(
                "SET channel order must exactly match the EEG-Emotion montage"
            )
        xs = []
        ys = []
        ts = []
        intervals = []
        for row in sorted(rows, key=lambda r: r["trial"]):
            start, end = interval(eeg, row["start_event"], row["end_event"])
            if any(start < b and end > a for a, b in intervals):
                raise ValueError("Overlapping retained trials")
            intervals.append((start, end))
            x = extract_de(
                np.asarray(data[:, start:end], dtype=np.float64) * 1e-6,
                int(eeg["srate"]),
                config["bands"],
                target_rate=config["target_rate"],
                window_seconds=config["window_seconds"],
                step_seconds=config["step_seconds"],
                filter_order=config["filter_order"],
            )
            xs.append(x)
            ys.extend([row["label"]] * len(x))
            ts.extend([row["trial"]] * len(x))
        name = f"sub{subject}_session{session}.npz"
        np.savez_compressed(
            output / name,
            x=np.concatenate(xs),
            y=np.asarray(ys, np.int64),
            trial=np.asarray(ts, np.int64),
        )
        items.append(dict(subject=subject, session=session, file=name))
        with fdt.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance.append(
            dict(
                subject=subject,
                session=session,
                set_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                fdt_sha256=digest,
                header_data_reference=str(eeg["data"]),
                actual_fdt=fdt.name,
            )
        )
        print(f"Extracted subject {subject}: {len(ys)} DE windows", flush=True)
    write_json(
        output / "manifest.json",
        dict(
            schema_version=1,
            feature="de",
            bands=config["bands"],
            normalized=False,
            classes=m["classes"],
            channels=montage["channels"],
            records=items,
        ),
    )
    write_json(
        output / "preprocessing.json",
        dict(config=config, trial_manifest=m, sources=provenance),
    )


def main():
    parser=argparse.ArgumentParser(description='Extract retained EEG-Emotion subjects numbered 1..20 from confirmed trials')
    parser.add_argument('--config',default=str(ROOT/'Configs/preprocess_eeglab.json'))
    args=parser.parse_args()
    c=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    extract(c)

#!/usr/bin/env python
"""Build native-grid aerosol features on six model levels.

CPU preparation reads HDF5 samples and writes a resumable feature cache.
Statistics are fitted on January-October samples or supplied explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.aerosol_vertical import (
    CHANNEL_NAMES, GROUP_NATIVE_INDICES, NATIVE_AEROSOL_NAMES, NORMALIZATION,
    SCHEMA, SELECTED_LEVEL_INDICES, UNIT_POLICY, PositiveChannelScales,
    TrainingMoments, audit_samples, group_selected_native9, is_training_date,
    log_transform, object_sha256, sha256_file, standardize,
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_source_unchanged(path: Path, record: dict) -> None:
    info = path.stat()
    if (info.st_size, info.st_mtime_ns) != (record["source_size"], record["source_mtime_ns"]):
        raise RuntimeError(f"Source changed after SHA-256 audit; rerun preparation without modifying source data: {path}")


def iter_selected_hours(path: Path, record: dict):
    check_source_unchanged(path, record)
    with h5py.File(path, "r") as handle:
        for hour in range(24):
            # Read the six selected levels.
            yield group_selected_native9(handle["aerosol_future"][hour, SELECTED_LEVEL_INDICES, :, :, :])
    check_source_unchanged(path, record)


def _normalization_identity(audit: dict) -> dict:
    return {"schema": SCHEMA, "normalization": NORMALIZATION,
            "selected_level_indices": SELECTED_LEVEL_INDICES, "channels": CHANNEL_NAMES,
            "groups": GROUP_NATIVE_INDICES, "unit_policy": UNIT_POLICY,
            "source_schema": audit["source_schema"], "source_grid": audit["source_grid"],
            "feature_code_sha256": sha256_file(ROOT / "src/data/aerosol_vertical.py"),
            "cache_builder_code_sha256": sha256_file(Path(__file__)),
            "training_source_sha256": {name: item["source_sha256"] for name, item in audit["files"].items() if is_training_date(item["sample_date"])} }


def fit_statistics(sample_dir: Path, output_dir: Path, audit: dict) -> dict:
    identity = _normalization_identity(audit)
    fingerprint = object_sha256(identity)
    if not identity["training_source_sha256"]:
        raise ValueError("No January-October samples are available. December-only local data can be audited but cannot fit training normalization")
    statistics_file = output_dir / "normalization_stats.json"
    if statistics_file.exists():
        saved = read_json(statistics_file)
        if saved.get("fit_identity_sha256") != fingerprint:
            raise ValueError("Existing normalization belongs to different training data/protocol; use a new output directory")
        return saved
    training = [(name, item) for name, item in audit["files"].items() if is_training_date(item["sample_date"])]
    scales = PositiveChannelScales()
    scale_progress_file = output_dir / "scale_fit_progress.json"
    processed: list[str] = []
    if scale_progress_file.exists():
        progress = read_json(scale_progress_file)
        if progress.get("fit_identity_sha256") != fingerprint:
            raise ValueError("Scale resume state belongs to different training data/protocol")
        scales.total = np.asarray(progress["total"], dtype=np.float64)
        scales.count = np.asarray(progress["count"], dtype=np.int64)
        processed = progress["completed_samples"]
    for index, (name, record) in enumerate(training, 1):
        if name in processed:
            continue
        for selected in iter_selected_hours(sample_dir / name, record):
            scales.update(selected, record["sample_date"])
        processed.append(name)
        atomic_json(scale_progress_file, {"fit_identity_sha256": fingerprint, "total": scales.total.tolist(),
                    "count": scales.count.tolist(), "completed_samples": processed})
        print(f"[scales {index}/{len(training)}] {name}", flush=True)
    scale_values = scales.finish()
    moments = TrainingMoments()
    moments_progress_file = output_dir / "log_stats_fit_progress.json"
    processed = []
    if moments_progress_file.exists():
        progress = read_json(moments_progress_file)
        if progress.get("fit_identity_sha256") != fingerprint or progress.get("scales_sha256") != object_sha256(scale_values):
            raise ValueError("Log-statistics resume state belongs to different training data/scales")
        moments.total = np.asarray(progress["total"], dtype=np.float64)
        moments.squared = np.asarray(progress["squared"], dtype=np.float64)
        moments.count = int(progress["count"])
        processed = progress["completed_samples"]
    for index, (name, record) in enumerate(training, 1):
        if name in processed:
            continue
        for selected in iter_selected_hours(sample_dir / name, record):
            moments.update(log_transform(selected, scale_values), record["sample_date"])
        processed.append(name)
        atomic_json(moments_progress_file, {"fit_identity_sha256": fingerprint, "scales_sha256": object_sha256(scale_values),
                    "total": moments.total.tolist(), "squared": moments.squared.tolist(), "count": moments.count,
                    "completed_samples": processed})
        print(f"[logstats {index}/{len(training)}] {name}", flush=True)
    saved = {**moments.finish(), **scale_values, "fit_identity_sha256": fingerprint, "fit_identity": identity,
             "training_sample_count": len(training), "schema": SCHEMA,
             "selected_level_indices": SELECTED_LEVEL_INDICES, "level_coordinate": "zero-based native model index; not physical altitude",
             "channels": CHANNEL_NAMES, "unit_policy": UNIT_POLICY}
    atomic_json(statistics_file, saved)
    return saved


def validate_cache_file(path: Path, expected_shape: list[int], expected_sha256: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"Cache SHA-256 mismatch: {path}; do not silently overwrite existing results")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != expected_shape or array.dtype != np.float16:
        raise ValueError(f"Invalid cache shape/dtype: {path}: {array.shape}/{array.dtype}")
    # Validate by hour without materializing the whole year's data.
    for hour in range(24):
        values = array[:, :, hour]
        if not np.isfinite(values).all() or np.any(np.abs(values) > 10):
            raise ValueError(f"Invalid cached normalized values: {path}, hour={hour + 1}")
    del array
    return expected_sha256 or sha256_file(path)


def write_cache_file(source: Path, record: dict, output: Path, shape: list[int], stats: dict) -> str:
    temporary = output.with_name(output.stem + ".tmp.npy")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=tuple(shape))
    try:
        for hour, selected in enumerate(iter_selected_hours(source, record)):
            array[:, :, hour] = standardize(log_transform(selected, stats), stats)
        array.flush()
    finally:
        del array
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    digest = validate_cache_file(temporary, shape)
    os.replace(temporary, output)
    return digest


def supplied_statistics(source: Path, output_dir: Path) -> dict:
    """Load and copy fixed aerosol normalization parameters."""
    saved = read_json(source)
    for key, expected in (("schema", SCHEMA), ("channels", CHANNEL_NAMES),
                          ("selected_level_indices", SELECTED_LEVEL_INDICES)):
        if saved.get(key) != expected:
            raise ValueError(f"Supplied normalization has an incompatible {key}")
    for key, shape in (("mean", (9, 6)), ("std", (9, 6)), ("channel_scales_native", (9,))):
        values = np.asarray(saved[key], dtype=np.float64)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid supplied normalization {key}")
        if key != "mean" and np.any(values <= 0):
            raise ValueError(f"Supplied normalization {key} must be positive")
    target = output_dir / "normalization_stats.json"
    content = source.read_bytes()
    if target.exists() and target.read_bytes() != content:
        raise ValueError("Output cache contains different normalization parameters; use a new directory")
    if not target.exists():
        target.write_bytes(content)
    return saved


def build_cache(sample_dir: Path, output_dir: Path, *, height: int = 149, width: int = 191,
                allow_subset: bool = False, audit_only: bool = False, reference_audit: Path | None = None,
                normalization_file: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".build.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Another cache preparation may be active: {lock}. If a previous process was killed, verify it has ended before removing ONLY this lock file") from exc
    os.close(descriptor)
    try:
        audit = audit_samples(sample_dir, height=height, width=width, allow_subset=allow_subset)
        if reference_audit is not None:
            reference = read_json(reference_audit)
            wanted = {name: row["source_sha256"] for name, row in reference["files"].items()}
            found = {name: row["source_sha256"] for name, row in audit["files"].items()}
            if wanted != found:
                raise ValueError("Server source SHA-256 list does not match the provided reference audit")
        atomic_json(output_dir / "sample_audit.json", audit)
        if audit_only:
            print(f"Source audit complete: {len(audit['files'])} files; formal={audit['formal_audit_qualified']}", flush=True)
            return audit
        stats = (supplied_statistics(normalization_file, output_dir) if normalization_file is not None
                 else fit_statistics(sample_dir, output_dir, audit))
        stats_sha256 = sha256_file(output_dir / "normalization_stats.json")
        shape = [9, 6, 24, height, width]
        manifest_file = output_dir / "manifest.json"
        identity = {"schema": SCHEMA, "channels": CHANNEL_NAMES, "selected_level_indices": SELECTED_LEVEL_INDICES,
                    "spatial_pool": 1, "shape_per_file": shape, "forecast_hours": 24,
                    "normalization_stats_sha256": stats_sha256,
                    "source_sha256": {name: row["source_sha256"] for name, row in audit["files"].items()}}
        fingerprint = object_sha256(identity)
        old = read_json(manifest_file) if manifest_file.exists() else None
        if old is not None and old.get("cache_identity_sha256") != fingerprint:
            raise ValueError("Existing cache manifest differs from the source data/normalization/protocol; choose a new cache directory")
        files = {} if old is None else old.get("files", {})
        manifest = {**identity, "status": "incomplete", "formal_audit_qualified": audit["formal_audit_qualified"],
                    "cache_identity_sha256": fingerprint, "cache_layout": "species,levels,future_hour,y,x",
                    "normalization": NORMALIZATION, "normalization_stats_file": "normalization_stats.json",
                    "sample_count": len(audit["files"]), "completed_count": len(files), "dtype": "float16",
                    "source_grid": [height, width], "unit_policy": UNIT_POLICY,
                    "channel_formulas": {channel: "+".join(NATIVE_AEROSOL_NAMES[i] for i in indices) for channel, indices in zip(CHANNEL_NAMES, GROUP_NATIVE_INDICES)},
                    "level_coordinate": "zero-based native model index",
                    "files": files}
        atomic_json(manifest_file, manifest)
        for index, (name, record) in enumerate(audit["files"].items(), 1):
            cache_file = output_dir / (Path(name).stem + ".npy")
            if name in files:
                digest = validate_cache_file(cache_file, shape, files[name]["cache_sha256"])
                print(f"[resume {index}/{len(audit['files'])}] {name}", flush=True)
            elif cache_file.exists():
                # A crash may occur after atomic rename but before manifest
                # commit. Recompute to a temporary file and require equality.
                recovered = cache_file.with_name(cache_file.stem + ".recover.npy")
                digest = write_cache_file(sample_dir / name, record, recovered, shape, stats)
                existing_digest = validate_cache_file(cache_file, shape)
                if digest != existing_digest:
                    raise ValueError(f"Untracked existing cache differs from recomputation: {cache_file}")
                recovered.unlink()
                print(f"[recovered {index}/{len(audit['files'])}] {name}", flush=True)
            else:
                digest = write_cache_file(sample_dir / name, record, cache_file, shape, stats)
                print(f"[build {index}/{len(audit['files'])}] {name}", flush=True)
            files[name] = {**record, "cache_file": cache_file.name, "cache_sha256": digest, "shape": shape}
            manifest["completed_count"] = len(files)
            atomic_json(manifest_file, manifest)
        manifest["status"] = "complete"
        manifest["all_selected_aerosol_hours_numeric_validated"] = True
        atomic_json(manifest_file, manifest)
        print(f"Complete native6 pool1 cache: {output_dir}; formal={manifest['formal_audit_qualified']}", flush=True)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-subset", action="store_true", help="Build a cache from a subset of initialization dates")
    parser.add_argument("--expected-height", type=int, default=149)
    parser.add_argument("--expected-width", type=int, default=191)
    parser.add_argument("--reference-audit", type=Path, help="Optional local sample_audit.json: require the exact same filenames and SHA-256")
    parser.add_argument("--normalization-file", type=Path, help="Apply supplied parameters instead of fitting training statistics")
    args = parser.parse_args()
    build_cache(args.sample_dir, args.output_dir, height=args.expected_height, width=args.expected_width,
                allow_subset=args.allow_subset, audit_only=args.audit_only, reference_audit=args.reference_audit,
                normalization_file=args.normalization_file)


if __name__ == "__main__":
    main()

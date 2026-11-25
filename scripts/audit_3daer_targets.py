#!/usr/bin/env python3
"""Read-only target availability audit; independent of the frozen training code.

Requires numpy and h5py, no torch/GPU. Reads only observation targets, masks,
longitude and time axes. Exit 2 means the target contract is not satisfied;
it does not attempt to repair data, change masks or start/resume training.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

NAME = re.compile(r"sample_(2025\d{4})12\.h5$")
FIELDS = ("sample", "split", "lead_hour", "target_time_utc", "raw_mask_count",
          "finite_truth_count", "valid_full_grid_count", "valid_east80_count",
          "truth_min_k", "truth_max_k", "empty_reason")


def split_for_date(stamp: str) -> str:
    return "train" if stamp < "20251101" else "val" if stamp < "20251201" else "test"


def expected_dates(split: str) -> list[str]:
    dates = [(datetime(2025, 1, 1) + timedelta(days=n)).strftime("%Y%m%d") for n in range(365)]
    return [d for d in dates if split == "all" or split_for_date(d) == split]


def decode_time(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def target_rows(path: Path, *, grid_shape=(149, 191)):
    found = NAME.fullmatch(path.name)
    if not found:
        raise ValueError("Not a daily 2025 12UTC source filename")
    stamp = found.group(1)
    init = datetime.strptime(stamp + "12", "%Y%m%d%H").replace(tzinfo=timezone.utc)
    before = path.stat()
    content = hashlib.sha256()
    with h5py.File(path, "r") as source:
        needed = ("bt13_future", "bt13_future_valid_mask", "grid_lon", "future_hours", "future_time_utc")
        missing = [key for key in needed if key not in source]
        if missing:
            raise ValueError(f"Missing required target fields: {missing}")
        arrays = {key: source[key][...] for key in needed}
    for key, values in arrays.items():
        content.update(key.encode())
        content.update(str(values.shape).encode())
        content.update(str(values.dtype).encode())
        if values.dtype.hasobject:
            content.update(json.dumps([decode_time(v) for v in values.ravel()]).encode())
        else:
            content.update(np.ascontiguousarray(values).tobytes())
    # Use the same float32 conversion and boolean mask as B13CascadeDataset.
    truth = arrays["bt13_future"].astype(np.float32)
    raw_mask = arrays["bt13_future_valid_mask"]
    lon = arrays["grid_lon"].astype(np.float32)
    times, hours = arrays["future_time_utc"], arrays["future_hours"]
    if truth.shape != (24, *grid_shape) or raw_mask.shape != truth.shape:
        raise ValueError(f"Unexpected target/mask shapes: {truth.shape}, {raw_mask.shape}")
    if lon.shape != grid_shape or not np.isfinite(lon).all():
        raise ValueError("Longitude grid missing, nonfinite or mismatched")
    if not np.isfinite(raw_mask).all():
        raise ValueError("Nonfinite target validity mask")
    if not np.array_equal(hours, np.arange(1, 25)) or times.shape != (24,):
        raise ValueError("Target time axes must contain 24 ordered leads")
    east = lon >= 80.0
    if not east.any():
        raise ValueError("Longitude grid contains no pixels east of 80 degrees E")
    mask = raw_mask.astype(bool)
    rows = []
    for i in range(24):
        text_time = decode_time(times[i])
        actual_time = datetime.fromisoformat(text_time.replace("Z", "+00:00"))
        if actual_time != init + timedelta(hours=i + 1):
            raise ValueError(f"Target time disagrees with lead {i+1}: {text_time}")
        finite = np.isfinite(truth[i])
        full = mask[i] & finite
        domain = full & east
        reason = ""
        if not domain.any():
            if not mask[i].any() and not finite.any():
                reason = "all_mask_false_and_all_truth_nonfinite"
            elif not mask[i].any():
                reason = "all_mask_false"
            elif not finite.any():
                reason = "all_truth_nonfinite"
            elif not full.any():
                reason = "no_finite_truth_under_observation_mask"
            else:
                reason = "no_valid_truth_east_of_80E"
        rows.append(dict(sample=path.name, split=split_for_date(stamp), lead_hour=i+1,
            target_time_utc=text_time, raw_mask_count=int(mask[i].sum()),
            finite_truth_count=int(finite.sum()), valid_full_grid_count=int(full.sum()),
            valid_east80_count=int(domain.sum()),
            truth_min_k=float(truth[i][finite].min()) if finite.any() else None,
            truth_max_k=float(truth[i][finite].max()) if finite.any() else None,
            empty_reason=reason))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed during the read-only audit")
    record = dict(path=str(path.resolve()), size_bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                  target_fields_sha256=content.hexdigest(), full_source_sha256=None)
    return rows, record


def audit(sample_dir: Path, output_dir: Path, *, split="all", grid_shape=(149, 191), progress=True):
    sample_dir, output_dir = sample_dir.resolve(), output_dir.resolve()
    if sample_dir == output_dir:
        raise ValueError("Write audit reports outside the source directory")
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)
    files = sorted(p for p in sample_dir.glob("sample_2025????12.h5")
                   if NAME.fullmatch(p.name) and
                   (split == "all" or split_for_date(NAME.fullmatch(p.name).group(1)) == split))
    dates = [NAME.fullmatch(p.name).group(1) for p in files]
    expected = expected_dates(split)
    all_rows, records, errors = [], {}, []
    for index, path in enumerate(files, 1):
        try:
            rows, record = target_rows(path, grid_shape=grid_shape)
            all_rows.extend(rows)
            records[path.name] = record
            empty = [r["lead_hour"] for r in rows if r["empty_reason"]]
            if progress:
                print(f"[{index}/{len(files)}] {path.name}: empty_leads={empty}", flush=True)
        except Exception as exc:
            error = dict(sample=path.name, error=f"{type(exc).__name__}: {exc}")
            errors.append(error)
            if progress:
                print(f"[{index}/{len(files)}] ERROR {path.name}: {error['error']}", flush=True)
    empty_rows = [row for row in all_rows if row["empty_reason"]]
    coverage_ok = dates == expected
    summary = dict(schema="diffcloud_target_availability_audit_v1", source_directory=str(sample_dir),
        requested_split=split, source_data_modified=False, code_or_cache_modified=False,
        trained_or_inferred=False, full_source_files_hashed=False,
        scope="bt13_future, observation mask, longitude, lead and UTC axes only; no aerosol reads",
        status="passed" if coverage_ok and not errors and not empty_rows else "failed_target_contract",
        expected_files=len(expected), found_files=len(files), date_coverage_complete=coverage_ok,
        missing_dates=sorted(set(expected)-set(dates)), unexpected_dates=sorted(set(dates)-set(expected)),
        audited_frames=len(all_rows), empty_frames=len(empty_rows), errors=errors,
        by_split={part:dict(audited_frames=sum(r["split"] == part for r in all_rows),
                           empty_frames=sum(r["split"] == part for r in empty_rows),
                           affected_samples=sorted({r["sample"] for r in empty_rows if r["split"] == part}))
                  for part in ("train", "val", "test")},
        empty_frame_details=empty_rows, source_records=records)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("frame_validity.csv", all_rows), ("empty_frames.csv", empty_rows)):
        with (output_dir/name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(values)
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    args = parser.parse_args()
    result = audit(args.sample_dir, args.output_dir, split=args.split)
    print(json.dumps({k:result[k] for k in ("status", "source_directory", "found_files", "audited_frames", "empty_frames", "missing_dates", "by_split", "errors")}, ensure_ascii=False, indent=2))
    print(f"Reports: {args.output_dir.resolve()}", flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()

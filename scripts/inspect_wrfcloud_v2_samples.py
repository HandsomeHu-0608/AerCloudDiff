from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_b13cascade import (
    WRF_CLOUD_LOG1P_NAMES,
    WRF_CLOUD_SIGNED_LOG1P_NAMES,
    WRF_CLOUD_V2_NAMES,
)


DATE_RE = re.compile(r"(2025\d{4})12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate wrfcloud-v2 B13 samples and compute train-only normalization.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--sample-glob", default="sample_2025????12.h5")
    parser.add_argument("--output-dir", default="reports/stage2_round3a_wrfcloud_v2_inspection")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--stats-end-date", default="20251031")
    parser.add_argument("--spatial-stride", type=int, default=8)
    return parser.parse_args()


def sample_date(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse YYYYMMDD12 from {path.name}")
    return match.group(1)


def decode_names(dataset) -> list[str]:
    out = []
    for value in dataset[()]:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            out.append(value.decode("utf-8", errors="strict"))
        else:
            out.append(str(value))
    return out


def transformed(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if name in WRF_CLOUD_SIGNED_LOG1P_NAMES:
        return (np.sign(values) * np.log1p(np.abs(values))).astype(np.float32)
    if name in WRF_CLOUD_LOG1P_NAMES:
        return np.log1p(np.maximum(values, 0.0)).astype(np.float32)
    return values


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    files = sorted(sample_dir.glob(args.sample_glob))
    if not files:
        files = sorted(sample_dir.rglob(args.sample_glob))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise FileNotFoundError(f"No samples matched {sample_dir / args.sample_glob}")

    stride = max(1, int(args.spatial_stride))
    sampled_values: dict[str, list[np.ndarray]] = {name: [] for name in WRF_CLOUD_V2_NAMES}
    errors: list[str] = []
    inspected = []
    stats_files = 0
    for index, path in enumerate(files, start=1):
        date = sample_date(path)
        entry = {"path": str(path), "date": date, "errors": []}
        try:
            with h5py.File(path, "r") as handle:
                for key in ("wrf_cloud_future", "wrf_cloud_var_names", "bt13_future", "future_hours"):
                    if key not in handle:
                        entry["errors"].append(f"missing {key}")
                if not entry["errors"]:
                    names = decode_names(handle["wrf_cloud_var_names"])
                    shape = tuple(handle["wrf_cloud_future"].shape)
                    target_shape = tuple(handle["bt13_future"].shape)
                    expected_shape = (24, 149, 191, len(WRF_CLOUD_V2_NAMES))
                    if names != WRF_CLOUD_V2_NAMES:
                        entry["errors"].append(f"wrf_cloud_var_names mismatch: {names}")
                    if shape != expected_shape:
                        entry["errors"].append(f"wrf_cloud_future shape {shape} != {expected_shape}")
                    if target_shape != (24, 149, 191):
                        entry["errors"].append(f"bt13_future shape {target_shape} != (24, 149, 191)")
                    hours = handle["future_hours"][...].astype(int).tolist()
                    if hours != list(range(1, 25)):
                        entry["errors"].append(f"future_hours {hours} != 1..24")
                    if not entry["errors"]:
                        cloud = handle["wrf_cloud_future"]
                        for channel, name in enumerate(names):
                            data = cloud[..., channel]
                            if not np.isfinite(data).all():
                                entry["errors"].append(f"{name} contains non-finite values")
                            if date <= args.stats_end_date:
                                values = data[:, ::stride, ::stride]
                                sampled_values[name].append(transformed(values, name).reshape(-1))
                        if date <= args.stats_end_date:
                            stats_files += 1
                    entry["shape"] = list(shape)
                    entry["names"] = names
        except Exception as exc:
            entry["errors"].append(str(exc))
        if entry["errors"]:
            errors.extend(f"{path.name}: {message}" for message in entry["errors"])
        inspected.append(entry)
        print(f"[{index}/{len(files)}] {path.name}: {'ERROR' if entry['errors'] else 'ok'}", flush=True)

    channels = {}
    for name in WRF_CLOUD_V2_NAMES:
        if not sampled_values[name]:
            errors.append(f"No training-stat values collected for {name}")
            continue
        values = np.concatenate(sampled_values[name]).astype(np.float64, copy=False)
        values = values[np.isfinite(values)]
        if values.size == 0:
            errors.append(f"No finite training-stat values collected for {name}")
            continue
        low, high = np.percentile(values, [0.1, 99.9])
        clipped = np.clip(values, low, high)
        std = float(clipped.std())
        channels[name] = {
            "transform": (
                "signed_log1p"
                if name in WRF_CLOUD_SIGNED_LOG1P_NAMES
                else "log1p"
                if name in WRF_CLOUD_LOG1P_NAMES
                else "identity"
            ),
            "clip_low": float(low),
            "clip_high": float(high),
            "mean": float(clipped.mean()),
            "std": max(std, 1.0e-6),
            "sampled_count": int(values.size),
            "raw_min": float(values.min()),
            "raw_max": float(values.max()),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization = {
        "schema": "wrfcloud_v2_normalization_v1",
        "sample_dir": str(sample_dir),
        "sample_glob": args.sample_glob,
        "stats_date_end": args.stats_end_date,
        "stats_files": stats_files,
        "spatial_stride": stride,
        "var_names": WRF_CLOUD_V2_NAMES,
        "channels": channels,
    }
    (output_dir / "wrfcloud_v2_normalization.json").write_text(
        json.dumps(normalization, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "schema": "wrfcloud_v2_sample_inspection_v1",
        "sample_count": len(files),
        "stats_files": stats_files,
        "error_count": len(errors),
        "errors": errors,
        "files": inspected,
    }
    (output_dir / "wrfcloud_v2_inspection.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# WRFCloud-v2 sample inspection",
        "",
        f"- Samples: `{len(files)}`",
        f"- Train-stat samples: `{stats_files}`",
        f"- Errors: `{len(errors)}`",
        f"- Expected shape: `[24,149,191,{len(WRF_CLOUD_V2_NAMES)}]`",
        f"- Spatial statistics stride: `{stride}`",
        "",
        "## Expected channel order",
        "",
        "`" + ", ".join(WRF_CLOUD_V2_NAMES) + "`",
    ]
    if errors:
        lines.extend(["", "## Errors", ""] + [f"- {message}" for message in errors])
    (output_dir / "wrfcloud_v2_inspection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote inspection outputs under {output_dir.resolve()}")
    if errors:
        raise SystemExit(f"WRFCloud-v2 inspection failed with {len(errors)} errors")


if __name__ == "__main__":
    main()

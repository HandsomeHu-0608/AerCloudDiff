"""Nine aerosol groups on six native WRF-Chem levels and the model grid."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date as calendar_date, timedelta
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from src.data.aerosol_fields import NATIVE_AEROSOL_NAMES, decode_strings

SCHEMA = "aerclouddiff_aerosol_native6_cache_v1"
CHANNEL_NAMES = ["SO4", "BC1", "BC2", "OC1", "OC2", "DUST_fine", "DUST_coarse", "SEAS_fine", "SEAS_coarse"]
SELECTED_LEVEL_INDICES = [0, 4, 9, 16, 24, 33]
GROUP_NATIVE_INDICES = [[0], [1], [2], [3], [4], [5, 6], [7, 8, 9], [10, 11], [12, 13]]
NORMALIZATION = "train_months_01_10_native_group_scale_log1p_channel_level_fixed"
UNIT_POLICY = "native units; common units within dust and sea-salt groups; positive channel mean scales from training samples"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sample_date(path: str | Path) -> str:
    match = re.search(r"_(2025\d{4})12\.h5$", Path(path).name)
    if match is None:
        raise ValueError(f"Expected a 2025 daily 12 UTC sample filename: {path}")
    value = match.group(1)
    return calendar_date(int(value[:4]), int(value[4:6]), int(value[6:])).isoformat()


def is_training_date(value: str) -> bool:
    return "2025-01-01" <= value <= "2025-10-31"


def _canonical_unit(unit: str) -> str:
    return str(unit).lower().replace("µ", "u").replace("μ", "u").replace(" ", "").replace("/kg", "kg-1").replace("^", "")


def audit_source_header(handle: h5py.File, height: int, width: int, *, full_2d_contract: bool = False) -> None:
    shapes = {"aerosol_future": (24, 34, height, width, 14),
              "aerosol_column_summary_future": (24, height, width, 28),
              "wrf_cloud_future": (24, height, width, 23)}
    for key, shape in shapes.items():
        if key not in handle or tuple(handle[key].shape) != shape:
            raise ValueError(f"{handle.filename}: {key} must have shape {shape}")
    if "aod550_future" not in handle or tuple(handle["aod550_future"].shape) != (24, height, width):
        raise ValueError("aod550_future must have shape (24, height, width)")
    if decode_strings(handle["aerosol_var_names"][...]) != NATIVE_AEROSOL_NAMES:
        raise ValueError("aerosol_var_names must follow the 14-component input order")
    expected_summary = [*(f"{name}_column_mean" for name in NATIVE_AEROSOL_NAMES), *(f"{name}_column_sum" for name in NATIVE_AEROSOL_NAMES)]
    if decode_strings(handle["aerosol_column_summary_var_names"][...]) != expected_summary:
        raise ValueError("aerosol_column_summary_var_names must follow the 28-channel input order")
    if not np.array_equal(handle["future_hours"][...], np.arange(1, 25)):
        raise ValueError("future_hours must be exactly 1..24")
    for key in ("grid_lat", "grid_lon"):
        if key not in handle or tuple(handle[key].shape) != (height, width) or not np.isfinite(handle[key][...]).all():
            raise ValueError(f"Missing or invalid source {key}")
    if "aerosol_var_units" in handle:
        units = decode_strings(handle["aerosol_var_units"][...])
        if len(units) != 14:
            raise ValueError("aerosol_var_units must cover 14 channels")
        for indices in GROUP_NATIVE_INDICES:
            if len({_canonical_unit(units[i]) for i in indices}) != 1:
                raise ValueError("Cannot sum native dust/sea-salt bins whose declared units differ")
    if full_2d_contract:
        from src.data.dataset_b13cascade import WRF_CLOUD_V2_NAMES
        required = {
            "bt13_hist": (25,height,width), "bt13_hist_valid_mask": (25,height,width),
            "cloud_hist_feat": (25,height,width,8), "cloud_hist_valid_mask": (25,height,width),
            "bt13_future": (24,height,width), "bt13_future_valid_mask": (24,height,width),
            "wrf_meteo_future": (24,height,width,6), "wrf_rad_future": (24,height,width,3),
            "coszen_hist": (25,height,width), "daylight_mask_hist": (25,height,width),
            "coszen_future": (24,height,width), "daylight_mask_future": (24,height,width),
            "k_wrf_future": (24,height,width), "k_wrf_valid_mask": (24,height,width),
            "static_feat": (height,width,4), "himawari_on_wrf_valid_mask": (height,width),
            "cloud_history_offsets_hours": (25,), "future_time_utc": (24,),
            **{k:(24,) for k in ("lead_time_norm","hour_utc_sin","hour_utc_cos","hour_bjt_sin","hour_bjt_cos")},
        }
        for key, shape in required.items():
            if key not in handle or handle[key].shape != shape:
                raise ValueError(f"Incomplete legacy 2D input/target contract: {handle.filename}: {key} != {shape}")
        if not np.array_equal(handle["cloud_history_offsets_hours"][...], np.arange(-24,1)):
            raise ValueError("History offsets must be -24..0")
        if decode_strings(handle["wrf_cloud_var_names"][...]) != WRF_CLOUD_V2_NAMES:
            raise ValueError("WRF-cloud23 names do not match the pretrained backbone")
        for key, count in (("cloud_feature_names",8),("meteo_var_names",6),("rad_var_names",3),("static_var_names",4)):
            if key not in handle or handle[key].shape != (count,):
                raise ValueError(f"Missing field-order metadata: {key}")
        init = str(handle.attrs.get("init_time_utc", ""))
        if not init.startswith(sample_date(handle.filename)+"T12:00:00"):
            raise ValueError("Filename and declared init_time_utc disagree")


def selected_native9(aerosol_hour: np.ndarray) -> np.ndarray:
    """Select fixed native levels, then group bins: [9,6,H,W], raw units."""
    value = np.asarray(aerosol_hour, dtype=np.float32)
    if value.ndim != 4 or value.shape[0] != 34 or value.shape[-1] != 14:
        raise ValueError("Expected one native aerosol hour [34,H,W,14]")
    return group_selected_native9(value[SELECTED_LEVEL_INDICES])


def group_selected_native9(selected_levels: np.ndarray) -> np.ndarray:
    """Group an HDF5 hyperslab containing only the six selected native levels."""
    value = np.asarray(selected_levels, dtype=np.float32)
    if value.ndim != 4 or value.shape[0] != 6 or value.shape[-1] != 14:
        raise ValueError("Expected six explicitly selected native levels [6,H,W,14]")
    if not np.isfinite(value).all() or np.any(value < -1.0e-12):
        raise ValueError("Non-finite or meaningfully negative selected aerosol values")
    value = np.maximum(value, 0.0)
    return np.stack([value[..., indices].sum(axis=-1, dtype=np.float32) for indices in GROUP_NATIVE_INDICES])


def _validate_selected(value: np.ndarray) -> None:
    if value.ndim != 4 or value.shape[:2] != (9, 6) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("Expected finite nonnegative native-grid features [9,6,H,W]")


class PositiveChannelScales:
    """One positive-value mean per grouped channel, fitted on training only."""
    def __init__(self) -> None:
        self.total = np.zeros(9, np.float64)
        self.count = np.zeros(9, np.int64)

    def update(self, selected: np.ndarray, date: str) -> None:
        if not is_training_date(date):
            raise ValueError("Scale fitting is restricted to January-October training samples")
        _validate_selected(selected)
        value = selected.astype(np.float64)
        self.total += value.sum(axis=(1, 2, 3))
        self.count += np.count_nonzero(value > 0, axis=(1, 2, 3))

    def finish(self) -> dict:
        if not np.any(self.count):
            raise ValueError("No positive training aerosol values available for normalization scales")
        scales = np.divide(self.total, self.count, out=np.ones(9), where=self.count > 0)
        return {"channel_scales_native": scales.tolist(), "positive_count_per_channel": self.count.tolist(),
                "scale_definition": "positive mean across training samples, selected levels, forecast hours and grid cells; all-zero channels use 1"}


def log_transform(selected: np.ndarray, scales: Mapping) -> np.ndarray:
    _validate_selected(selected)
    scale = np.asarray(scales["channel_scales_native"], dtype=np.float64)
    if scale.shape != (9,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid fixed native-unit channel scales")
    return np.log1p(selected.astype(np.float64) / scale[:, None, None, None]).astype(np.float32)


class TrainingMoments:
    """Streaming channel/native-level statistics; validation/test forbidden."""
    def __init__(self) -> None:
        self.count = 0
        self.total = np.zeros((9, 6), np.float64)
        self.squared = np.zeros((9, 6), np.float64)

    def update(self, transformed: np.ndarray, date: str) -> None:
        if not is_training_date(date):
            raise ValueError("Normalization fitting is restricted to January-October training samples")
        _validate_selected(transformed)
        value = transformed.astype(np.float64)
        self.total += value.sum(axis=(-2, -1))
        self.squared += np.square(value).sum(axis=(-2, -1))
        self.count += int(value.shape[-2] * value.shape[-1])

    def finish(self) -> dict:
        if self.count == 0:
            raise ValueError("No training samples available for fitting normalization")
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.squared / self.count - mean * mean, 0))
        std = np.maximum(std, 1.0e-6)
        return {"mean": mean.tolist(), "std": std.tolist(), "count_per_channel_level": self.count,
                "normalization": NORMALIZATION, "training_date_range": ["2025-01-01", "2025-10-31"]}


def standardize(transformed: np.ndarray, stats: Mapping) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.shape != (9, 6) or std.shape != (9, 6) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid fixed training normalization")
    values = (transformed - mean[..., None, None]) / std[..., None, None]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite standardized aerosol cache values")
    return np.clip(values, -10, 10).astype(np.float16)


def audit_samples(sample_dir: str | Path, *, height: int = 149, width: int = 191,
                  allow_subset: bool = False, progress: bool = True) -> dict:
    paths = sorted(Path(sample_dir).glob("sample_2025????12.h5"))
    if not paths:
        raise FileNotFoundError(f"No legacy v2 samples under {sample_dir}")
    dates = [sample_date(path) for path in paths]
    required = [(calendar_date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    formal = dates == required
    if not allow_subset and not formal:
        missing = sorted(set(required) - set(dates))
        raise ValueError(f"Formal cache requires all 365 unique 2025 dates; found={len(paths)}, missing={missing[:12]}")
    files = {}
    schema_signature = None
    for index, path in enumerate(paths, 1):
        stat = path.stat()
        if stat.st_size == 0:
            raise ValueError(f"Empty source file: {path}")
        with h5py.File(path, "r") as handle:
            audit_source_header(handle, height, width, full_2d_contract=not allow_subset)
            signature = {key: decode_strings(handle[key][...]) for key in ("aerosol_var_names", "aerosol_column_summary_var_names", "wrf_cloud_var_names")}
            if "aerosol_var_units" in handle:
                signature["aerosol_var_units"] = decode_strings(handle["aerosol_var_units"][...])
            if not allow_subset:
                for key in ("cloud_feature_names","meteo_var_names","rad_var_names","static_var_names"):
                    signature[key] = decode_strings(handle[key][...])
            if schema_signature is None:
                schema_signature = signature
            elif signature != schema_signature:
                raise ValueError(f"Field names or units changed within the source collection: {path.name}")
        fingerprint = sha256_file(path)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while being audited: {path}")
        files[path.name] = {"sample_date": sample_date(path), "source_size": stat.st_size,
                            "source_mtime_ns": stat.st_mtime_ns, "source_sha256": fingerprint}
        if progress:
            print(f"[audit {index}/{len(paths)}] {path.name}", flush=True)
    return {"schema": SCHEMA, "formal_audit_qualified": formal and not allow_subset and (height, width) == (149, 191),
            "sample_count": len(paths), "source_grid": [height, width], "files": files,
            "source_schema": schema_signature, "unit_policy": UNIT_POLICY,
            "audit_scope": "readable HDF5 headers, legacy fields/shapes/names, units if supplied, time/grid, file SHA-256; selected aerosol numeric validation occurs during all cache build hours"}

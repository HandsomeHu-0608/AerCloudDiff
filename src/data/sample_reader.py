from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

SUPPORTED_EXTENSIONS = {
    ".npz",
    ".npy",
    ".h5",
    ".hdf5",
    ".nc",
    ".netcdf",
    ".mat",
}


@dataclass(frozen=True)
class FieldInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str


def decode_array(values: np.ndarray) -> list[Any]:
    out: list[Any] = []
    for value in np.asarray(values):
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            out.append(value.decode("utf-8"))
        elif hasattr(value, "item"):
            item = value.item()
            if isinstance(item, (bytes, bytearray)):
                out.append(item.decode("utf-8"))
            else:
                out.append(item)
        else:
            out.append(value)
    return out


def discover_sample_files(sample_dir: str | Path, recursive: bool = True) -> list[Path]:
    sample_dir = Path(sample_dir)
    iterator = sample_dir.rglob("*") if recursive else sample_dir.glob("*")
    return sorted([p for p in iterator if p.is_file()])


def supported_sample_files(sample_dir: str | Path, recursive: bool = True) -> list[Path]:
    return [p for p in discover_sample_files(sample_dir, recursive) if p.suffix.lower() in SUPPORTED_EXTENSIONS]


def extension_counts(paths: Iterable[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        suffix = path.suffix.lower() or "<none>"
        counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def parse_init_time_from_name(path: Path) -> str | None:
    match = re.search(r"(20\d{6})(\d{2})", path.stem)
    if not match:
        return None
    ymd, hour = match.groups()
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}T{hour}:00:00Z"


def parse_datetime_from_name(path: Path) -> datetime | None:
    text = parse_init_time_from_name(path)
    if text is None:
        return None
    try:
        return datetime.strptime(text.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def filter_files_by_date(
    files: Iterable[Path],
    start: str | None = None,
    end: str | None = None,
) -> list[Path]:
    if not start and not end:
        return list(files)
    start_dt = datetime.strptime(start, "%Y-%m-%d") if start else None
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else None
    out: list[Path] = []
    for path in files:
        dt = parse_datetime_from_name(path)
        if dt is None:
            continue
        if start_dt is not None and dt < start_dt:
            continue
        if end_dt is not None and dt > end_dt.replace(hour=23, minute=59, second=59):
            continue
        out.append(path)
    return out


def h5_top_level_schema(path: str | Path) -> list[FieldInfo]:
    infos: list[FieldInfo] = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                infos.append(FieldInfo(key, tuple(obj.shape), str(obj.dtype), "dataset"))
            elif isinstance(obj, h5py.Group):
                infos.append(FieldInfo(key, tuple(), "group", "group"))
    return infos


def h5_all_dataset_schema(path: str | Path) -> list[FieldInfo]:
    infos: list[FieldInfo] = []
    with h5py.File(path, "r") as f:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                infos.append(FieldInfo(name, tuple(obj.shape), str(obj.dtype), "dataset"))
        f.visititems(visitor)
    return infos


def read_name_list(h5_file: h5py.File, key: str) -> list[str]:
    if key not in h5_file:
        return []
    return [str(x) for x in decode_array(h5_file[key][()])]


def get_cloud_channel_indices(h5_file: h5py.File) -> dict[str, int]:
    names = read_name_list(h5_file, "cloud_feature_names")
    return {name: i for i, name in enumerate(names)}


def read_file_index(path: str | Path) -> list[Path]:
    path = Path(path)
    files: list[Path] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = row.get("path") or row.get("file") or row.get("full_path")
            if value:
                files.append(Path(value))
    return files


def write_file_index(sample_dir: str | Path, output_csv: str | Path) -> list[Path]:
    sample_dir = Path(sample_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    files = supported_sample_files(sample_dir, recursive=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "path",
            "filename",
            "suffix",
            "size_bytes",
            "init_time_utc",
            "schema_version",
            "has_cloud_hist_feat",
            "has_cloud_future_feat",
            "has_wrf_meteo_future",
            "has_wrf_rad_future",
            "has_aerosol_future",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in files:
            row: dict[str, Any] = {
                "path": str(path),
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "init_time_utc": parse_init_time_from_name(path),
                "schema_version": "",
                "has_cloud_hist_feat": "",
                "has_cloud_future_feat": "",
                "has_wrf_meteo_future": "",
                "has_wrf_rad_future": "",
                "has_aerosol_future": "",
            }
            if path.suffix.lower() in {".h5", ".hdf5"}:
                try:
                    with h5py.File(path, "r") as h5:
                        row["init_time_utc"] = h5.attrs.get("init_time_utc", row["init_time_utc"])
                        row["schema_version"] = h5.attrs.get("sample_schema_version", "")
                        for key in [
                            "cloud_hist_feat",
                            "cloud_future_feat",
                            "wrf_meteo_future",
                            "wrf_rad_future",
                            "aerosol_future",
                        ]:
                            row[f"has_{key}"] = str(key in h5)
                except Exception as exc:
                    row["schema_version"] = f"ERROR: {exc}"
            writer.writerow(row)
    return files


def evenly_spaced(paths: list[Path], max_files: int | None) -> list[Path]:
    if max_files is None or max_files <= 0 or len(paths) <= max_files:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, max_files).round().astype(int)
    seen: set[int] = set()
    selected: list[Path] = []
    for idx in indices:
        if int(idx) not in seen:
            selected.append(paths[int(idx)])
            seen.add(int(idx))
    return selected

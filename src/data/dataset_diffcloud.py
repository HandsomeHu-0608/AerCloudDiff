from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data.sample_reader import filter_files_by_date, read_file_index, supported_sample_files
from src.data.transforms import channelwise_standardize, clean_finite, log1p_cot, normalize_log_cot, sincos_encode

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover - exercised when torch is missing.
    torch = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class DiffCloudDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str = "train", files: list[Path] | None = None):
        if torch is None:
            raise ImportError(f"PyTorch is required for DiffCloudDataset: {_TORCH_IMPORT_ERROR}")

        dataset_cfg = config.get("dataset", config)
        self.sample_dir = Path(dataset_cfg["sample_dir"])
        self.file_index = dataset_cfg.get("file_index")
        self.sample_glob = dataset_cfg.get("sample_glob", "*.h5")
        self.max_files = dataset_cfg.get("max_files")
        self.split_cfg = dataset_cfg.get("splits", {}).get(split, {})
        self.history_channels = list(dataset_cfg.get("history_channels", ["cloud_binary", "CLOT"]))
        self.use_all_history_cloud_channels = bool(dataset_cfg.get("use_all_history_cloud_channels", False))
        self.use_future_wrf_condition = dataset_cfg.get("use_future_wrf_condition", "auto")
        self.condition_mode = dataset_cfg.get("condition_mode")
        self.aerosol_condition_mode = dataset_cfg.get("aerosol_condition_mode", "none")
        self.include_so4_future = bool(dataset_cfg.get("include_so4_future", False))
        self.cot_clip_max = float(dataset_cfg.get("cot_clip_max", 200.0))
        self.patch_size = dataset_cfg.get("patch_size")
        self.patch_sampling = dataset_cfg.get("patch_sampling")
        self.split = split
        self.seed = int(dataset_cfg.get("seed", config.get("training", {}).get("seed", 42)))
        self.patches_per_file = int(dataset_cfg.get("patches_per_file", 1))
        if split != "train":
            self.patches_per_file = 1
        self.patches_per_file = max(1, self.patches_per_file)
        if self.condition_mode is None:
            if self.use_future_wrf_condition is False or self.use_future_wrf_condition == "none":
                self.condition_mode = "history_only"
            elif self.aerosol_condition_mode == "full_3d_flatten":
                self.condition_mode = "aerosol_full3d_patch"
            else:
                self.condition_mode = "wrf_meteo_rad_aod"
        if self.patch_sampling is None:
            self.patch_sampling = "random" if split == "train" and self.patch_size is not None else "center"
        elif split != "train" and self.patch_sampling == "random":
            self.patch_sampling = "center"

        if files is not None:
            all_files = list(files)
        elif self.file_index and Path(self.file_index).exists():
            all_files = read_file_index(self.file_index)
        else:
            all_files = sorted(self.sample_dir.rglob(self.sample_glob))
            if not all_files:
                all_files = supported_sample_files(self.sample_dir)
        all_files = [p for p in all_files if p.suffix.lower() in {".h5", ".hdf5"}]
        start = self.split_cfg.get("start", dataset_cfg.get("date_start"))
        end = self.split_cfg.get("end", dataset_cfg.get("date_end"))
        all_files = filter_files_by_date(all_files, start=start, end=end)
        if self.max_files is not None:
            all_files = all_files[: int(self.max_files)]
        if not all_files:
            raise FileNotFoundError(f"No HDF5 sample files found under {self.sample_dir}")
        self.files = all_files

        self._last_shapes: dict[str, tuple[int, ...]] = {}

    def __len__(self) -> int:
        return len(self.files) * self.patches_per_file

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx % len(self.files)]
        try:
            with h5py.File(path, "r") as f:
                return self._read_h5(path, f, idx)
        except Exception as exc:
            raise RuntimeError(f"Failed reading sample file {path}: {exc}") from exc

    def _read_h5(self, path: Path, f: h5py.File, idx: int) -> dict[str, Any]:
        required = ["cloud_hist_feat", "cloud_future_feat"]
        for key in required:
            if key not in f:
                raise KeyError(f"{path}: missing required key {key}")

        cloud_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["cloud_feature_names"][()]]
        name_to_idx = {name: i for i, name in enumerate(cloud_names)}
        for key in ["cloud_binary", "CLOT"]:
            if key not in name_to_idx:
                raise KeyError(f"{path}: cloud_feature_names missing {key}; found {cloud_names}")

        y_slice, x_slice = self._spatial_slices(f["cloud_future_feat"].shape[1:3], idx=idx)
        future = f["cloud_future_feat"][:, y_slice, x_slice, :]
        hist = f["cloud_hist_feat"][:, y_slice, x_slice, :]
        if future.ndim != 4 or hist.ndim != 4:
            raise ValueError(f"{path}: expected cloud arrays [T,H,W,C], got hist={hist.shape}, future={future.shape}")

        cb_idx = name_to_idx["cloud_binary"]
        cot_idx = name_to_idx["CLOT"]
        future_mask = self._future_valid_mask(f, future, y_slice, x_slice)

        cloud_probability = np.clip(clean_finite(future[..., cb_idx]), 0.0, 1.0)
        raw_cot = future[..., cot_idx].astype(np.float32)
        valid_cot = np.isfinite(raw_cot) & (raw_cot >= 0)
        log_cot = normalize_log_cot(log1p_cot(raw_cot, self.cot_clip_max), self.cot_clip_max)

        y = np.stack([cloud_probability, clean_finite(log_cot)], axis=0)  # [C,T,H,W]
        valid_mask = future_mask.astype(np.float32)[None, ...]
        cloudy_mask = (future_mask & valid_cot & (cloud_probability >= 0.5)).astype(np.float32)[None, ...]
        loss_mask = np.concatenate([valid_mask, cloudy_mask], axis=0)

        condition = self._build_condition(f, hist, name_to_idx, future.shape[0], y_slice, x_slice)
        sample = {
            "condition": torch.from_numpy(np.ascontiguousarray(condition, dtype=np.float32)),
            "target": torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=np.float32)),
            "cloudy_mask": torch.from_numpy(np.ascontiguousarray(cloudy_mask, dtype=np.float32)),
            "loss_mask": torch.from_numpy(np.ascontiguousarray(loss_mask, dtype=np.float32)),
            "path": str(path),
        }
        return sample

    def _future_valid_mask(self, f: h5py.File, future: np.ndarray, y_slice: slice, x_slice: slice) -> np.ndarray:
        if "cloud_future_valid_mask" in f:
            mask = f["cloud_future_valid_mask"][:, y_slice, x_slice].astype(bool)
        else:
            mask = np.isfinite(future).all(axis=-1)
        if mask.shape != future.shape[:3]:
            raise ValueError(f"cloud_future_valid_mask shape {mask.shape} does not match future {future.shape[:3]}")
        return mask

    def _build_condition(
        self,
        f: h5py.File,
        hist: np.ndarray,
        name_to_idx: dict[str, int],
        t_future: int,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        h, w = hist.shape[1], hist.shape[2]
        parts: list[np.ndarray] = []

        if self.use_all_history_cloud_channels:
            hist_channels = []
            for c in range(hist.shape[-1]):
                data = clean_finite(hist[..., c])
                if c == name_to_idx.get("CLOT"):
                    data = normalize_log_cot(log1p_cot(data, self.cot_clip_max), self.cot_clip_max)
                hist_channels.append(data)
        else:
            hist_channels = []
            for name in self.history_channels:
                idx = name_to_idx.get(name)
                if idx is None:
                    continue
                data = hist[..., idx]
                if name == "CLOT":
                    data = normalize_log_cot(log1p_cot(data, self.cot_clip_max), self.cot_clip_max)
                else:
                    data = clean_finite(data)
                hist_channels.append(data)
        if not hist_channels:
            raise ValueError("No valid history cloud channels selected")
        hist_flat = np.stack(hist_channels, axis=0)  # [C_hist,T_hist,H,W]
        hist_flat = hist_flat.reshape(hist_flat.shape[0] * hist_flat.shape[1], h, w)
        hist_repeated = np.repeat(hist_flat[:, None, :, :], t_future, axis=1)
        parts.append(hist_repeated)

        if self._should_use_future_wrf(f):
            if self._include_meteo():
                meteo_keys = ["wrf_meteo_future"]
            else:
                meteo_keys = []
            if self._include_radiation():
                meteo_keys.append("wrf_rad_future")
            for key in meteo_keys:
                if key in f:
                    arr = f[key][:, y_slice, x_slice, :]  # [T,H,W,C]
                    parts.append(np.moveaxis(arr, -1, 0))
            if self._include_aod_k():
                for key in ["aod550_future", "k_wrf_future"]:
                    if key in f:
                        parts.append(f[key][:, y_slice, x_slice][None, ...])
            if self._include_static() and "static_feat" in f:
                static = f["static_feat"][y_slice, x_slice, :]  # [H,W,C]
                static = np.moveaxis(static, -1, 0)
                static = np.repeat(static[:, None, :, :], t_future, axis=1)
                parts.append(static)
            if self.condition_mode == "aerosol_column_summary" and "aerosol_future" in f:
                aerosol = f["aerosol_future"][:, :, y_slice, x_slice, :]  # [T,Z,H,W,A]
                col_mean = np.nanmean(aerosol, axis=1)
                col_sum = np.nansum(aerosol, axis=1)
                aerosol_summary = np.concatenate([col_mean, col_sum], axis=-1)  # [T,H,W,2A]
                parts.append(np.moveaxis(aerosol_summary, -1, 0))
            if self.condition_mode == "aerosol_full3d_patch" and "aerosol_future" in f:
                aerosol = f["aerosol_future"][:, :, y_slice, x_slice, :]  # [T,Z,H,W,A]
                aerosol = np.moveaxis(aerosol, -1, 1)  # [T,A,Z,H,W]
                aerosol = aerosol.reshape(t_future, -1, aerosol.shape[-2], aerosol.shape[-1])
                parts.append(np.moveaxis(aerosol, 0, 1))  # [A*Z,T,H,W]
            if self.include_so4_future and "so4_future" in f:
                so4 = f["so4_future"][:, :, y_slice, x_slice]  # [T,Z,H,W]
                so4 = np.moveaxis(so4, 0, 1)
                parts.append(so4)

        parts.append(self._time_condition(f, t_future, h, w))
        condition = np.concatenate(parts, axis=0)
        return channelwise_standardize(condition)

    def _spatial_slices(self, shape_hw: tuple[int, int], idx: int = 0) -> tuple[slice, slice]:
        h, w = int(shape_hw[0]), int(shape_hw[1])
        if self.patch_size is None:
            return slice(None), slice(None)
        if isinstance(self.patch_size, int):
            ph = pw = int(self.patch_size)
        else:
            ph, pw = int(self.patch_size[0]), int(self.patch_size[1])
        ph = min(ph, h)
        pw = min(pw, w)
        if self.patch_sampling == "random" and (h > ph or w > pw):
            y_rng = max(1, h - ph + 1)
            x_rng = max(1, w - pw + 1)
            y0 = int(np.random.randint(0, y_rng))
            x0 = int(np.random.randint(0, x_rng))
        else:
            y0 = max(0, (h - ph) // 2)
            x0 = max(0, (w - pw) // 2)
        return slice(y0, y0 + ph), slice(x0, x0 + pw)

    def _should_use_future_wrf(self, f: h5py.File) -> bool:
        if self.condition_mode == "history_only":
            return False
        if self.use_future_wrf_condition is True:
            return True
        if self.use_future_wrf_condition is False or self.use_future_wrf_condition == "none":
            return False
        return any(key in f for key in ["wrf_meteo_future", "wrf_rad_future", "aerosol_future", "aod550_future"])

    def _include_meteo(self) -> bool:
        return self.condition_mode in {
            "wrf_meteo",
            "wrf_meteo_rad",
            "wrf_meteo_rad_aod",
            "aerosol_column_summary",
            "aerosol_full3d_patch",
        }

    def _include_radiation(self) -> bool:
        return self.condition_mode in {
            "wrf_meteo_rad",
            "wrf_meteo_rad_aod",
            "aerosol_column_summary",
            "aerosol_full3d_patch",
        }

    def _include_aod_k(self) -> bool:
        return self.condition_mode in {
            "wrf_meteo_rad_aod",
            "aerosol_column_summary",
            "aerosol_full3d_patch",
        }

    def _include_static(self) -> bool:
        return self.condition_mode in {
            "wrf_meteo_rad_aod",
            "aerosol_column_summary",
            "aerosol_full3d_patch",
        }

    def _time_condition(self, f: h5py.File, t_future: int, h: int, w: int) -> np.ndarray:
        if "future_hours" in f:
            lead = f["future_hours"][()].astype(np.float32)
        else:
            lead = np.arange(1, t_future + 1, dtype=np.float32)
        lead_norm = lead / max(float(np.nanmax(lead)), 1.0)
        lead_sin, lead_cos = sincos_encode(lead, period=24.0)

        hour = np.zeros(t_future, dtype=np.float32)
        doy = np.zeros(t_future, dtype=np.float32)
        if "future_time_utc" in f:
            from datetime import datetime

            raw_times = f["future_time_utc"][()]
            for i, raw in enumerate(raw_times):
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                try:
                    dt = datetime.strptime(text.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
                    hour[i] = dt.hour
                    doy[i] = dt.timetuple().tm_yday
                except Exception:
                    hour[i] = 0.0
                    doy[i] = 0.0
        hour_sin, hour_cos = sincos_encode(hour, period=24.0)
        doy_sin, doy_cos = sincos_encode(np.maximum(doy, 1.0), period=366.0)
        channels = np.stack([lead_norm, lead_sin, lead_cos, hour_sin, hour_cos, doy_sin, doy_cos], axis=0)
        return np.repeat(channels[:, :, None, None], h, axis=2).repeat(w, axis=3).astype(np.float32)

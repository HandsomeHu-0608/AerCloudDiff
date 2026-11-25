from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data.sample_reader import filter_files_by_date, read_file_index, supported_sample_files
from src.data.aerosol_fields import AEROSOL_CHANNEL_NAMES
from src.data.transforms import channelwise_standardize, clean_finite, log1p_cot, normalize_log_cot

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # pragma: no cover - exercised when torch is missing.
    torch = None
    Dataset = object
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


FORBIDDEN_CONDITION_KEYS = {"bt13_future", "cloud_future_feat", "cloud_future_target_core"}

WRF_CLOUD_V2_NAMES = [
    "TSK",
    "CLDFRA_LOW",
    "CLDFRA_MID",
    "CLDFRA_HIGH",
    "LWP",
    "FROZEN_WP",
    "CTOP_HGT",
    "CTOP_TEMP",
    "RH_LOW",
    "RH_MID",
    "RH_HIGH",
    "WMAX_LOW",
    "WMAX_MID",
    "WMAX_HIGH",
    "U700",
    "V700",
    "U300",
    "V300",
    "LWUPT",
    "LWCRE",
    "MAXMF",
    "PREC_CU_1H",
    "PREC_GRID_1H",
]

CONDITION_GROUP_IDS = {
    "history_b13": 0,
    "wrf_regular": 1,
    "static_time": 2,
}

WRF_CLOUD_LOG1P_NAMES = {"LWP", "FROZEN_WP", "MAXMF", "PREC_CU_1H", "PREC_GRID_1H"}
WRF_CLOUD_SIGNED_LOG1P_NAMES = {"MAXMF"}


class B13CascadeDataset(Dataset):
    """Band-13 8-hour cascade dataset for [B,C,T,H,W] diffusion models."""

    def __init__(self, config: dict[str, Any], split: str = "train", files: list[Path] | None = None):
        if torch is None:
            raise ImportError(f"PyTorch is required for B13CascadeDataset: {_TORCH_IMPORT_ERROR}")

        dataset_cfg = config.get("dataset", config)
        self.dataset_cfg = dataset_cfg
        self.sample_dir = Path(dataset_cfg["sample_dir"])
        self.file_index = dataset_cfg.get("file_index")
        self.sample_glob = dataset_cfg.get("sample_glob", "*.h5")
        self.max_files = dataset_cfg.get("max_files")
        self.split_cfg = (dataset_cfg.get("splits") or {}).get(split, {})
        self.split = split
        self.seed = int(dataset_cfg.get("seed", config.get("training", {}).get("seed", 42)))

        self.target_key = str(dataset_cfg.get("target_key", "bt13_future"))
        self.target_valid_mask_key = str(dataset_cfg.get("target_valid_mask_key", "bt13_future_valid_mask"))
        self.history_key = str(dataset_cfg.get("history_key", "bt13_hist"))
        self.history_valid_mask_key = str(dataset_cfg.get("history_valid_mask_key", "bt13_hist_valid_mask"))
        self.cascade_horizon = int(dataset_cfg.get("cascade_horizon", 8))
        configured_blocks = int(dataset_cfg.get("cascade_blocks", 3))
        configured_starts = dataset_cfg.get("cascade_start_hours")
        if configured_starts is None:
            configured_starts = [block * self.cascade_horizon for block in range(configured_blocks)]
        self.cascade_start_hours = [int(value) for value in configured_starts]
        if not self.cascade_start_hours or self.cascade_start_hours[0] != 0:
            raise ValueError("dataset.cascade_start_hours must start at 0")
        if self.cascade_start_hours != sorted(set(self.cascade_start_hours)):
            raise ValueError("dataset.cascade_start_hours must be strictly increasing")
        self.cascade_blocks = len(self.cascade_start_hours)
        if configured_blocks != self.cascade_blocks:
            raise ValueError(
                f"dataset.cascade_blocks={configured_blocks} does not match "
                f"cascade_start_hours length={self.cascade_blocks}"
            )
        self.forecast_hours = int(
            dataset_cfg.get("forecast_hours", self.cascade_start_hours[-1] + self.cascade_horizon)
        )
        if self.cascade_start_hours[-1] + self.cascade_horizon != self.forecast_hours:
            raise ValueError(
                "The final cascade window must end exactly at dataset.forecast_hours; "
                f"got start={self.cascade_start_hours[-1]}, horizon={self.cascade_horizon}, "
                f"forecast_hours={self.forecast_hours}"
            )
        blend_weights = dataset_cfg.get("cascade_blend_weights", [1.0] * self.cascade_horizon)
        self.cascade_blend_weights = np.asarray(blend_weights, dtype=np.float32)
        if self.cascade_blend_weights.shape != (self.cascade_horizon,) or np.any(self.cascade_blend_weights <= 0.0):
            raise ValueError(
                f"dataset.cascade_blend_weights must contain {self.cascade_horizon} positive values"
            )
        self.history_length = int(dataset_cfg.get("history_length", 25))
        self.teacher_forcing = bool(dataset_cfg.get("teacher_forcing", True))
        self.history_teacher_noise_std_k = float(dataset_cfg.get("history_teacher_noise_std_k", 0.0) or 0.0)
        self.history_teacher_drop_prob = float(dataset_cfg.get("history_teacher_drop_prob", 0.0) or 0.0)
        self.history_teacher_drop_prob = min(max(self.history_teacher_drop_prob, 0.0), 1.0)
        self.history_teacher_bias_std_k = float(dataset_cfg.get("history_teacher_bias_std_k", 0.0) or 0.0)
        self.history_teacher_blur_prob = float(dataset_cfg.get("history_teacher_blur_prob", 0.0) or 0.0)
        self.history_teacher_blur_prob = min(max(self.history_teacher_blur_prob, 0.0), 1.0)
        block_scale = dataset_cfg.get("history_teacher_block_scale")
        if block_scale is None:
            block_scale = [1.0] * self.cascade_blocks
        self.history_teacher_block_scale = [float(x) for x in block_scale]

        self.west_edge_weighting = bool(dataset_cfg.get("west_edge_weighting", False))
        # Optional training-only scoring domain.  Leave observation masks intact
        # for inference/history; legacy configurations keep their original loss.
        self.loss_min_lon = dataset_cfg.get("loss_min_lon")
        self.loss_min_lon = None if self.loss_min_lon is None else float(self.loss_min_lon)
        self.west_edge_lon_threshold = float(dataset_cfg.get("west_edge_lon_threshold", 80.0))
        self.west_edge_transition_lon = float(dataset_cfg.get("west_edge_transition_lon", 90.0))
        self.west_edge_weight = float(dataset_cfg.get("west_edge_weight", 1.35))
        self.west_edge_transition_weight = float(dataset_cfg.get("west_edge_transition_weight", 1.15))

        self.include_bt13_hist = bool(dataset_cfg.get("include_bt13_hist", True))
        # Round3f-Full physical anchors.  Keep both disabled by default so that
        # every pre-Round3f configuration retains its exact condition layout.
        self.include_bt13_hist_absolute = bool(
            dataset_cfg.get("include_bt13_hist_absolute", False)
        )
        self.include_absolute_lead_time = bool(
            dataset_cfg.get("include_absolute_lead_time", False)
        )
        self.include_cloud_hist = bool(dataset_cfg.get("include_cloud_hist", True))
        self.include_coszen = bool(dataset_cfg.get("include_coszen", True))
        self.include_daylight_mask = bool(dataset_cfg.get("include_daylight_mask", True))
        self.include_wrf_meteo = bool(dataset_cfg.get("include_wrf_meteo", True))
        self.include_wrf_rad = bool(dataset_cfg.get("include_wrf_rad", True))
        self.include_aod550 = bool(dataset_cfg.get("include_aod550", True))
        self.aod550_key = str(dataset_cfg.get("aod550_key", "aod550_future"))
        self.require_aod550 = bool(dataset_cfg.get("require_aod550", False))
        self.include_k_wrf = bool(dataset_cfg.get("include_k_wrf", True))
        self.include_aerosol_column_summary = bool(dataset_cfg.get("include_aerosol_column_summary", True))
        selected_summary_names = dataset_cfg.get("aerosol_column_summary_selected_names")
        self.aerosol_column_summary_selected_names = (
            [str(value) for value in selected_summary_names]
            if selected_summary_names is not None
            else None
        )
        self.require_aerosol_column_summary = bool(
            dataset_cfg.get("require_aerosol_column_summary", False)
        )
        self.include_aerosol_full3d = bool(dataset_cfg.get("include_aerosol_full3d", False))
        self.aerosol_full3d_expected_species = int(dataset_cfg.get("aerosol_full3d_expected_species", 14))
        self.aerosol_full3d_expected_levels = int(dataset_cfg.get("aerosol_full3d_expected_levels", 34))
        self.aerosol_full3d_cache_dir = dataset_cfg.get("aerosol_full3d_cache_dir")
        self.aerosol_full3d_cache_dir = (
            Path(self.aerosol_full3d_cache_dir) if self.aerosol_full3d_cache_dir else None
        )
        self.require_aerosol_full3d_cache = bool(
            dataset_cfg.get("require_aerosol_full3d_cache", False)
        )
        self.aerosol_full3d_cache_require_manifest = bool(
            dataset_cfg.get("aerosol_full3d_cache_require_manifest", False)
        )
        self.aerosol_full3d_cache_require_formal_audit = bool(
            dataset_cfg.get("aerosol_full3d_cache_require_formal_audit", False)
        )
        self.aerosol_full3d_cache_spatial_pool = int(
            dataset_cfg.get("aerosol_full3d_cache_spatial_pool", 1)
        )
        self.aerosol_full3d_cache_schema = str(
            dataset_cfg.get("aerosol_full3d_cache_schema", "diffcloud_aerosol3d_pooled_cache_v1")
        )
        self._aerosol_full3d_cache_layout = "cascade_blocks"
        self.aerosol_full3d_cache_lru_files = max(
            1, int(dataset_cfg.get("aerosol_full3d_cache_lru_files", 8))
        )
        if self.aerosol_full3d_cache_spatial_pool < 1:
            raise ValueError("dataset.aerosol_full3d_cache_spatial_pool must be >= 1")
        self._aerosol_full3d_cache_arrays: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.include_wrf_cloud = bool(dataset_cfg.get("include_wrf_cloud", False))
        self.require_wrf_cloud = bool(dataset_cfg.get("require_wrf_cloud", self.include_wrf_cloud))
        expected_cloud_names = dataset_cfg.get("wrf_cloud_expected_names", WRF_CLOUD_V2_NAMES)
        self.wrf_cloud_expected_names = [str(value) for value in expected_cloud_names]
        expected_spatial_shape = dataset_cfg.get("wrf_cloud_expected_spatial_shape", [149, 191])
        self.wrf_cloud_expected_spatial_shape = tuple(int(value) for value in expected_spatial_shape)
        if len(self.wrf_cloud_expected_spatial_shape) != 2:
            raise ValueError("dataset.wrf_cloud_expected_spatial_shape must contain [H,W]")
        self.wrf_cloud_normalization_file = dataset_cfg.get("wrf_cloud_normalization_file")
        self.wrf_cloud_normalization = self._load_wrf_cloud_normalization()
        self.last_wrf_cloud_condition_names: list[str] = []
        self.include_static = bool(dataset_cfg.get("include_static", True))
        self.include_time = bool(dataset_cfg.get("include_time", True))
        self.include_future_bt13_as_condition = bool(dataset_cfg.get("include_future_bt13_as_condition", False))
        if self.include_future_bt13_as_condition:
            raise ValueError("include_future_bt13_as_condition must stay false for forecast-realistic B13 cascade training")

        self.normalize_b13 = bool(dataset_cfg.get("normalize_b13", True))
        clip = dataset_cfg.get("b13_clip", [180.0, 330.0])
        self.b13_clip_min = float(clip[0])
        self.b13_clip_max = float(clip[1])
        if not self.b13_clip_max > self.b13_clip_min:
            raise ValueError(f"Invalid b13_clip={clip}; max must be greater than min")
        self.cot_clip_max = float(dataset_cfg.get("cot_clip_max", 150.0))

        self.predicted_history_cache_dir = dataset_cfg.get("predicted_history_cache_dir")
        self.predicted_history_cache_dir = (
            Path(self.predicted_history_cache_dir) if self.predicted_history_cache_dir else None
        )
        self.require_predicted_history_cache = bool(dataset_cfg.get("require_predicted_history_cache", False))
        self.predicted_history_require_manifest = bool(
            dataset_cfg.get("predicted_history_require_manifest", False)
        )
        self.predicted_history_start_epoch = int(dataset_cfg.get("predicted_history_start_epoch", 6))
        self.predicted_history_full_epoch = int(dataset_cfg.get("predicted_history_full_epoch", 15))
        self.predicted_history_max_probability = float(dataset_cfg.get("predicted_history_max_probability", 0.0))
        self.predicted_history_max_probability = min(max(self.predicted_history_max_probability, 0.0), 1.0)
        self.sample_audit_manifest = dataset_cfg.get("sample_audit_manifest")
        self.require_sample_audit_manifest = bool(
            dataset_cfg.get("require_sample_audit_manifest", False)
        )
        self.aerosol_provenance_manifest = dataset_cfg.get("aerosol_provenance_manifest")
        self.require_aerosol_provenance = bool(dataset_cfg.get("require_aerosol_provenance", False))
        self.sample_audit_require_reference_match = bool(
            dataset_cfg.get("sample_audit_require_reference_match", False)
        )
        self._validated_sample_audit_records: dict[str, dict[str, Any]] = {}
        self.current_epoch = 1

        self.patch_size = dataset_cfg.get("patch_size")
        self.patch_sampling = dataset_cfg.get("patch_sampling")
        self.patches_per_file = int(dataset_cfg.get("patches_per_file", 1))
        if split != "train":
            self.patches_per_file = 1
        self.patches_per_file = max(1, self.patches_per_file)
        if self.patch_sampling is None:
            self.patch_sampling = "random" if split == "train" and self.patch_size is not None else "center"
        elif split != "train" and self.patch_sampling == "random":
            self.patch_sampling = "center"

        if files is not None:
            all_files = list(files)
        elif self.file_index and Path(self.file_index).exists():
            all_files = read_file_index(self.file_index)
        else:
            all_files = sorted(self.sample_dir.glob(self.sample_glob))
            if not all_files:
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
            raise FileNotFoundError(f"No HDF5 B13 cascade sample files found under {self.sample_dir}")
        self.files = all_files
        self.last_condition_names: list[str] = []
        if self.require_sample_audit_manifest:
            self._validated_sample_audit_records = self._validate_external_manifest(
                self.sample_audit_manifest,
                expected_schema="aerosol_sample_audit_v1",
                label="sample audit",
                require_passed=True,
            )
        if self.require_aerosol_provenance:
            provenance_records = self._validate_external_manifest(
                self.aerosol_provenance_manifest,
                expected_schema="stage3_aer_aerosol_provenance_v1",
                label="Aerosol provenance",
                require_passed=False,
            )
            if self._validated_sample_audit_records:
                mismatched = [
                    sample.name
                    for sample in self.files
                    if str(provenance_records[sample.name].get("sample_sha256", "")).lower()
                    != str(self._validated_sample_audit_records[sample.name].get("sha256", "")).lower()
                ]
                if mismatched:
                    raise RuntimeError(
                        "Aerosol provenance/sample-audit SHA-256 mismatch; "
                        f"first samples: {mismatched[:5]}"
                    )
        if self.split == "train" and self.require_predicted_history_cache:
            self._validate_predicted_history_cache_files()
        if self.include_aerosol_full3d and self.require_aerosol_full3d_cache:
            self._validate_aerosol_full3d_cache_files()

    def __len__(self) -> int:
        return len(self.files) * self.cascade_blocks * self.patches_per_file

    def _validate_external_manifest(
        self,
        path_value: str | None,
        *,
        expected_schema: str,
        label: str,
        require_passed: bool,
    ) -> dict[str, dict[str, Any]]:
        if not path_value:
            raise ValueError(f"dataset.{label.replace(' ', '_')}_manifest is required")
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} manifest not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != expected_schema:
            raise RuntimeError(
                f"Unsupported {label} manifest schema {payload.get('schema')!r}; "
                f"expected {expected_schema!r}"
            )
        if require_passed and not bool(payload.get("passed", False)):
            raise RuntimeError(f"Required {label} manifest is not marked passed: {path}")
        records = payload.get("files")
        if not isinstance(records, list):
            raise RuntimeError(f"Required {label} manifest has no files list: {path}")
        by_name = {str(record.get("sample_name", "")): record for record in records}
        missing = [sample.name for sample in self.files if sample.name not in by_name]
        if missing:
            raise RuntimeError(
                f"Required {label} manifest does not cover {len(missing)} selected samples; "
                f"first missing: {missing[:5]}"
            )
        if expected_schema == "aerosol_sample_audit_v1":
            if bool(payload.get("structure_only", True)):
                raise RuntimeError(f"Formal sample audit must include numerical checks: {path}")
            if not bool(payload.get("sha256_enabled", False)):
                raise RuntimeError(f"Formal sample audit must include SHA-256: {path}")
            if not bool(payload.get("provenance_required", False)):
                raise RuntimeError(f"Formal sample audit must verify Aerosol provenance: {path}")
            if self.sample_audit_require_reference_match and not bool(
                payload.get("reference_manifest_verified", False)
            ):
                raise RuntimeError(
                    f"Formal sample audit did not match the local reference manifest: {path}"
                )
            invalid_audit = []
            for sample in self.files:
                record = by_name[sample.name]
                digest = str(record.get("sha256", "")).lower()
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    invalid_audit.append(f"{sample.name}:sha256")
                if int(record.get("size_bytes", -1)) != sample.stat().st_size:
                    invalid_audit.append(f"{sample.name}:size_bytes")
            if invalid_audit:
                raise RuntimeError(
                    f"Required sample audit contains stale/invalid records; first: {invalid_audit[:5]}"
                )
        if expected_schema == "stage3_aer_aerosol_provenance_v1":
            invalid = []
            attested_basis = (
                payload.get("provenance_basis")
                == "project_owner_attestation_and_sample_file_lineage"
            )
            if attested_basis:
                declaration_digest = str(payload.get("identity_declaration_sha256", "")).lower()
                if not bool(payload.get("passed", False)):
                    invalid.append("manifest:not_passed")
                if len(declaration_digest) != 64 or any(
                    char not in "0123456789abcdef" for char in declaration_digest
                ):
                    invalid.append("manifest:identity_declaration_sha256")
                identity = payload.get("identity", {})
                if str(identity.get("attestation_status", "")).lower() != "complete":
                    invalid.append("manifest:attestation_status")
                for sample in self.files:
                    record = by_name[sample.name]
                    if (
                        str(record.get("vertical_correction_status", "")).lower()
                        != "complete_user_attested"
                    ):
                        invalid.append(sample.name)
                    sample_digest = str(record.get("sample_sha256", "")).lower()
                    if len(sample_digest) != 64 or any(
                        char not in "0123456789abcdef" for char in sample_digest
                    ):
                        invalid.append(f"{sample.name}:sample_sha256")
                    if str(record.get("identity_declaration_sha256", "")).lower() != declaration_digest:
                        invalid.append(f"{sample.name}:identity_declaration_sha256")
            else:
                for sample in self.files:
                    record = by_name[sample.name]
                    if str(record.get("vertical_correction_status", "")).lower() != "complete":
                        invalid.append(sample.name)
                    for key in ("sample_sha256", "daily_extract_sha256", "source_provenance_sha256"):
                        value = str(record.get(key, "")).lower()
                        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                            invalid.append(f"{sample.name}:{key}")
            if invalid:
                raise RuntimeError(
                    f"Required Aerosol provenance contains invalid records; first: {invalid[:5]}"
                )
        return by_name

    def _aerosol_full3d_cache_path(self, sample_path: Path) -> Path:
        if self.aerosol_full3d_cache_dir is None:
            raise ValueError("aerosol_full3d_cache_dir is unset")
        return self.aerosol_full3d_cache_dir / f"{sample_path.stem}.npy"

    def _validate_aerosol_full3d_cache_files(self) -> None:
        if self.aerosol_full3d_cache_dir is None:
            raise ValueError(
                "include_aerosol_full3d with require_aerosol_full3d_cache=true requires "
                "dataset.aerosol_full3d_cache_dir"
            )
        manifest_path = self.aerosol_full3d_cache_dir / "manifest.json"
        if self.aerosol_full3d_cache_require_manifest and not manifest_path.is_file():
            raise RuntimeError(f"Missing complete 3-D aerosol cache manifest: {manifest_path}")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != self.aerosol_full3d_cache_schema:
                raise RuntimeError(f"Unsupported 3-D aerosol cache manifest schema: {manifest_path}")
            if int(manifest.get("spatial_pool", -1)) != self.aerosol_full3d_cache_spatial_pool:
                raise RuntimeError(
                    "3-D aerosol cache spatial_pool mismatch: "
                    f"manifest={manifest.get('spatial_pool')}, "
                    f"config={self.aerosol_full3d_cache_spatial_pool}"
                )
            if self.aerosol_full3d_cache_schema == "aerclouddiff_aerosol_native6_cache_v1":
                from src.utils.b13_checkpoint import checkpoint_fingerprint

                expected_channels = ["SO4", "BC1", "BC2", "OC1", "OC2", "DUST_fine", "DUST_coarse", "SEAS_fine", "SEAS_coarse"]
                expected_indices = [0, 4, 9, 16, 24, 33]
                if manifest.get("status") != "complete":
                    raise RuntimeError("Native-six aerosol cache is not complete")
                if self.aerosol_full3d_cache_require_formal_audit and not manifest.get("formal_audit_qualified", False):
                    raise RuntimeError("Native-six aerosol cache requires the complete annual audit")
                if manifest.get("channels") != expected_channels or manifest.get("selected_level_indices") != expected_indices:
                    raise RuntimeError("Native-six aerosol cache species/layer contract mismatch")
                expected_shape = [9, 6, self.forecast_hours, *self.wrf_cloud_expected_spatial_shape]
                if list(manifest.get("shape_per_file", [])) != expected_shape or self.aerosol_full3d_cache_spatial_pool != 1:
                    raise RuntimeError("Native-six aerosol cache must retain the full horizontal grid")
                if (self.aerosol_full3d_expected_species, self.aerosol_full3d_expected_levels) != (9, 6):
                    raise RuntimeError("Native-six aerosol configuration requires 9 species and 6 layers")
                stats_path = self.aerosol_full3d_cache_dir / "normalization_stats.json"
                if not stats_path.is_file() or checkpoint_fingerprint(stats_path) != manifest.get("normalization_stats_sha256"):
                    raise RuntimeError("Native-six aerosol normalization fingerprint mismatch")
                records = manifest.get("files", {})
                for sample in self.files:
                    record = records.get(sample.name)
                    if not isinstance(record, dict):
                        raise RuntimeError(f"Native-six cache does not cover {sample.name}")
                    stat = sample.stat()
                    if stat.st_size != record.get("source_size") or stat.st_mtime_ns != record.get("source_mtime_ns"):
                        raise RuntimeError(f"Source sample changed after cache construction: {sample}")
                # The preparation command verifies all source/cache SHA-256 values;
                # rank-local dataset construction uses stat checks, not 4x annual hashing.
                self._aerosol_full3d_cache_layout = "full_forecast"
            if self.aerosol_full3d_cache_schema == "diffcloud_aerosol3d_agl_cache_v2":
                if self.aerosol_full3d_cache_require_formal_audit and not bool(
                    manifest.get("formal_audit_qualified", False)
                ):
                    raise RuntimeError(
                        f"Aerosol cache was not built from a formal qualified audit: {manifest_path}"
                    )
                layout = str(manifest.get("cache_layout", ""))
                if layout != "[channel,level,time,y_pool,x_pool]":
                    raise RuntimeError(
                        f"Unsupported Aerosol cache layout {layout!r} in {manifest_path}"
                    )
                if int(manifest.get("forecast_hours", -1)) != self.forecast_hours:
                    raise RuntimeError(
                        "Aerosol cache forecast_hours mismatch: "
                        f"manifest={manifest.get('forecast_hours')}, config={self.forecast_hours}"
                    )
                if manifest.get("channels") != AEROSOL_CHANNEL_NAMES:
                    raise RuntimeError(
                        f"Aerosol cache channel contract mismatch in {manifest_path}"
                    )
                expected_shape = [
                    self.aerosol_full3d_expected_species,
                    self.aerosol_full3d_expected_levels,
                    self.forecast_hours,
                ]
                shape_per_file = list(manifest.get("shape_per_file", []))
                if shape_per_file[:3] != expected_shape:
                    raise RuntimeError(
                        f"Aerosol cache shape contract {shape_per_file} does not start with {expected_shape}"
                    )
                records = manifest.get("files", [])
                manifest_names = {
                    str(record.get("sample_name", "")) for record in records if isinstance(record, dict)
                }
                uncovered = [sample.name for sample in self.files if sample.name not in manifest_names]
                if uncovered:
                    raise RuntimeError(
                        f"Aerosol cache manifest does not cover selected samples: {uncovered[:5]}"
                    )
                self._aerosol_full3d_cache_layout = "full_forecast"
        missing = [path.name for path in self.files if not self._aerosol_full3d_cache_path(path).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"3-D aerosol cache is incomplete under {self.aerosol_full3d_cache_dir}; "
                f"first missing: {preview}"
            )

    def _read_aerosol_full3d_cache(
        self,
        sample_path: Path,
        block: int,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray:
        cache_path = self._aerosol_full3d_cache_path(sample_path)
        cache = self._aerosol_full3d_cache_arrays.pop(cache_path, None)
        if cache is None:
            cache = np.load(cache_path, mmap_mode="r")
            if self.aerosol_full3d_cache_schema == "aerclouddiff_aerosol_native6_cache_v1":
                expected = (9, 6, self.forecast_hours, *self.wrf_cloud_expected_spatial_shape)
                if cache.shape != expected or cache.dtype != np.float16:
                    raise RuntimeError(f"Native-six cache shape/dtype mismatch: {cache_path}")
        self._aerosol_full3d_cache_arrays[cache_path] = cache
        while len(self._aerosol_full3d_cache_arrays) > self.aerosol_full3d_cache_lru_files:
            self._aerosol_full3d_cache_arrays.popitem(last=False)
        if self._aerosol_full3d_cache_layout == "full_forecast":
            expected_prefix = (
                self.aerosol_full3d_expected_species,
                self.aerosol_full3d_expected_levels,
                self.forecast_hours,
            )
            if tuple(cache.shape[:3]) != expected_prefix:
                raise RuntimeError(
                    f"Unexpected 3-D aerosol cache shape {tuple(cache.shape)} in {cache_path}; "
                    f"expected prefix {expected_prefix}"
                )
            start = self.cascade_start_hours[int(block)]
            cached_block = cache[:, :, start : start + self.cascade_horizon]
        else:
            expected_prefix = (
                self.cascade_blocks,
                self.aerosol_full3d_expected_species,
                self.aerosol_full3d_expected_levels,
                self.cascade_horizon,
            )
            if tuple(cache.shape[:4]) != expected_prefix:
                raise RuntimeError(
                    f"Unexpected 3-D aerosol cache shape {tuple(cache.shape)} in {cache_path}; "
                    f"expected prefix {expected_prefix}"
                )
            cached_block = cache[int(block)]
        # Rollout/evaluation uses patch_size=None and therefore requests the
        # complete source grid as slice(None).  Its cached pooled grid should
        # be returned intact and is upsampled by the model before fusion.
        if (
            y_slice.start is None
            and y_slice.stop is None
            and x_slice.start is None
            and x_slice.stop is None
        ):
            return np.asarray(cached_block, dtype=np.float32)
        pool = self.aerosol_full3d_cache_spatial_pool
        y_start = int(y_slice.start or 0)
        x_start = int(x_slice.start or 0)
        y0 = y_start // pool
        x0 = x_start // pool
        target_h = int((int(y_slice.stop) - y_start + pool - 1) // pool)
        target_w = int((int(x_slice.stop) - x_start + pool - 1) // pool)
        if self._aerosol_full3d_cache_layout == "full_forecast" and pool != 1:
            cached = self._fractional_pooled_crop(
                cached_block,
                y_start=y_start,
                x_start=x_start,
                target_h=target_h,
                target_w=target_w,
                pool=pool,
            )
        else:
            cached = cached_block[:, :, :, y0 : y0 + target_h, x0 : x0 + target_w]
        if cached.shape[-2:] != (target_h, target_w):
            raise RuntimeError(
                f"3-D aerosol cache crop is out of range for {cache_path}: "
                f"got {cached.shape[-2:]}, expected {(target_h, target_w)}"
            )
        return np.asarray(cached, dtype=np.float32)

    @staticmethod
    def _fractional_pooled_crop(
        values: np.ndarray,
        *,
        y_start: int,
        x_start: int,
        target_h: int,
        target_w: int,
        pool: int,
    ) -> np.ndarray:
        """Sample a pooled global cache at the exact origin of a full-resolution patch.

        An odd patch origin lies halfway between two pool-2 cells.  Bilinear
        sampling removes the half-cell shift that integer division introduces.
        """

        values = np.asarray(values)
        y = (float(y_start) / float(pool)) + np.arange(target_h, dtype=np.float64)
        x = (float(x_start) / float(pool)) + np.arange(target_w, dtype=np.float64)
        if y[-1] > values.shape[-2] - 1 or x[-1] > values.shape[-1] - 1:
            raise RuntimeError(
                "3-D aerosol cache fractional crop is out of range: "
                f"requested y={y[0]:.3f}..{y[-1]:.3f}, x={x[0]:.3f}..{x[-1]:.3f}, "
                f"cache_hw={values.shape[-2:]}"
            )
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        y1 = np.minimum(y0 + 1, values.shape[-2] - 1)
        x1 = np.minimum(x0 + 1, values.shape[-1] - 1)
        wy = (y - y0).astype(np.float32)
        wx = (x - x0).astype(np.float32)
        top = values[..., y0[:, None], x0[None, :]] * (1.0 - wx)[None, None, None, None, :]
        top += values[..., y0[:, None], x1[None, :]] * wx[None, None, None, None, :]
        bottom = values[..., y1[:, None], x0[None, :]] * (1.0 - wx)[None, None, None, None, :]
        bottom += values[..., y1[:, None], x1[None, :]] * wx[None, None, None, None, :]
        return top * (1.0 - wy)[None, None, None, :, None] + bottom * wy[None, None, None, :, None]

    @property
    def target_length(self) -> int:
        return self.cascade_horizon

    @property
    def target_channels(self) -> int:
        return 1

    @property
    def wrf_cloud_channels(self) -> int:
        return len(self.wrf_cloud_expected_names) if self.include_wrf_cloud else 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = max(1, int(epoch))

    def predicted_history_probability(self) -> float:
        if self.split != "train" or self.predicted_history_max_probability <= 0.0:
            return 0.0
        if self.current_epoch < self.predicted_history_start_epoch:
            return 0.0
        if self.predicted_history_full_epoch <= self.predicted_history_start_epoch:
            return self.predicted_history_max_probability
        progress = (self.current_epoch - self.predicted_history_start_epoch + 1) / float(
            self.predicted_history_full_epoch - self.predicted_history_start_epoch + 1
        )
        return self.predicted_history_max_probability * min(max(progress, 0.0), 1.0)

    def _load_wrf_cloud_normalization(self) -> dict[str, Any] | None:
        if not self.include_wrf_cloud:
            return None
        if not self.wrf_cloud_normalization_file:
            raise ValueError("dataset.wrf_cloud_normalization_file is required when include_wrf_cloud=true")
        path = Path(self.wrf_cloud_normalization_file)
        if not path.is_file():
            raise FileNotFoundError(f"WRF cloud normalization file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = [str(value) for value in payload.get("var_names", [])]
        if names != self.wrf_cloud_expected_names:
            raise ValueError(
                f"WRF cloud normalization names {names} do not match expected {self.wrf_cloud_expected_names}"
            )
        channels = payload.get("channels", {})
        missing = [name for name in self.wrf_cloud_expected_names if name not in channels]
        if missing:
            raise ValueError(f"WRF cloud normalization is missing channels: {missing}")
        return payload

    def _validate_predicted_history_cache_files(self) -> None:
        if self.predicted_history_cache_dir is None:
            raise ValueError("require_predicted_history_cache=true but predicted_history_cache_dir is unset")
        missing = [path.name for path in self.files if not self._predicted_history_cache_path(path).is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Predicted-history cache is missing {len(missing)} training files under "
                f"{self.predicted_history_cache_dir}; first missing: {preview}"
            )
        manifest_path = self.predicted_history_cache_dir / "manifest.json"
        if self.predicted_history_require_manifest and not manifest_path.is_file():
            raise FileNotFoundError(f"Predicted-history cache manifest is required: {manifest_path}")
        if manifest_path.is_file():
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(manifest.get("split", "train")) != "train":
                raise ValueError(f"Predicted-history training cache must have split=train: {manifest_path}")
            expected = manifest.get("expected_files")
            if expected is not None and int(expected) != len(self.files):
                raise ValueError(
                    f"Predicted-history cache manifest expected_files={expected}, dataset has {len(self.files)}"
                )

    def _predicted_history_cache_path(self, sample_path: Path) -> Path:
        if self.predicted_history_cache_dir is None:
            raise ValueError("predicted_history_cache_dir is unset")
        return self.predicted_history_cache_dir / f"{sample_path.stem}.npz"

    def _read_predicted_history_cache(
        self,
        sample_path: Path,
        future_needed: int,
        y_slice: slice,
        x_slice: slice,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_path = self._predicted_history_cache_path(sample_path)
        if not cache_path.is_file():
            if self.require_predicted_history_cache:
                raise FileNotFoundError(f"Missing predicted-history cache: {cache_path}")
            raise FileNotFoundError(f"Predicted-history cache selected but unavailable: {cache_path}")
        with np.load(cache_path) as cache:
            if "pred_k" not in cache or "valid" not in cache:
                raise KeyError(f"{cache_path}: expected pred_k and valid arrays")
            pred = np.asarray(cache["pred_k"])
            valid = np.asarray(cache["valid"]).astype(bool)
        if pred.shape != (self.forecast_hours, *valid.shape[-2:]) or valid.shape != pred.shape:
            raise ValueError(f"{cache_path}: invalid cache shapes pred={pred.shape}, valid={valid.shape}")
        future = pred[:future_needed, y_slice, x_slice].astype(np.float32)
        future_mask = valid[:future_needed, y_slice, x_slice] & np.isfinite(future)
        return future, future_mask

    def _build_wrf_cloud_condition(
        self,
        f: h5py.File,
        start: int,
        end: int,
        y_slice: slice,
        x_slice: slice,
    ) -> tuple[np.ndarray | None, list[str]]:
        if not self.include_wrf_cloud:
            return None, []
        if "wrf_cloud_future" not in f or "wrf_cloud_var_names" not in f:
            raise KeyError("wrfcloud-v2 condition requested but required datasets are missing")
        names = self._read_names(f, "wrf_cloud_var_names")
        if names != self.wrf_cloud_expected_names:
            raise ValueError(f"wrf_cloud_var_names {names} != expected {self.wrf_cloud_expected_names}")
        values = f["wrf_cloud_future"][start:end, y_slice, x_slice, :].astype(np.float32)
        if values.ndim != 4 or values.shape[-1] != len(names):
            raise ValueError(f"wrf_cloud_future block must be [T,H,W,23], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("wrf_cloud_future contains non-finite values")
        if self.wrf_cloud_normalization is None:
            raise ValueError("WRF cloud normalization was not loaded")
        channels = []
        for index, name in enumerate(names):
            arr = values[..., index]
            stats = self.wrf_cloud_normalization["channels"][name]
            transform = str(stats.get("transform", "identity"))
            if transform == "log1p":
                arr = np.log1p(np.maximum(arr, 0.0))
            elif transform == "signed_log1p":
                arr = np.sign(arr) * np.log1p(np.abs(arr))
            elif transform != "identity":
                raise ValueError(f"Unsupported WRF cloud transform {transform!r} for {name}")
            low = float(stats["clip_low"])
            high = float(stats["clip_high"])
            mean = float(stats["mean"])
            std = max(float(stats["std"]), 1.0e-6)
            arr = np.clip(arr, low, high)
            arr = np.clip((arr - mean) / std, -6.0, 6.0)
            channels.append(arr.astype(np.float32))
        return np.moveaxis(np.stack(channels, axis=-1), -1, 0), [f"wrf_cloud_{name}" for name in names]

    def normalize_bt13(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if not self.normalize_b13:
            return clean_finite(arr)
        arr = np.clip(arr, self.b13_clip_min, self.b13_clip_max)
        arr = (arr - self.b13_clip_min) / (self.b13_clip_max - self.b13_clip_min)
        return clean_finite(arr)

    def denormalize_bt13(self, values: np.ndarray, clip: bool = True) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if not self.normalize_b13:
            return arr
        if clip:
            arr = np.clip(arr, 0.0, 1.0)
        return arr * (self.b13_clip_max - self.b13_clip_min) + self.b13_clip_min

    def __getitem__(self, idx: int) -> dict[str, Any]:
        file_stride = self.cascade_blocks * self.patches_per_file
        file_index = (idx // file_stride) % len(self.files)
        within_file = idx % file_stride
        block = within_file // self.patches_per_file
        patch_id = within_file % self.patches_per_file
        return self.read_block(file_index=file_index, block=block, patch_id=patch_id, global_idx=idx)

    def read_block(
        self,
        file_index: int,
        block: int,
        patch_id: int = 0,
        global_idx: int | None = None,
        history_override: np.ndarray | None = None,
        history_mask_override: np.ndarray | None = None,
        y_slice: slice | None = None,
        x_slice: slice | None = None,
    ) -> dict[str, Any]:
        path = self.files[int(file_index)]
        try:
            with h5py.File(path, "r") as f:
                if y_slice is None or x_slice is None:
                    y_slice, x_slice = self._spatial_slices(f[self.target_key].shape[1:3], idx=global_idx or patch_id)
                return self._read_h5(
                    path=path,
                    f=f,
                    block=int(block),
                    y_slice=y_slice,
                    x_slice=x_slice,
                    history_override=history_override,
                    history_mask_override=history_mask_override,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed reading B13 cascade sample file {path}: {exc}") from exc

    def _read_h5(
        self,
        path: Path,
        f: h5py.File,
        block: int,
        y_slice: slice,
        x_slice: slice,
        history_override: np.ndarray | None = None,
        history_mask_override: np.ndarray | None = None,
    ) -> dict[str, Any]:
        self._validate_required_keys(path, f)
        if block < 0 or block >= self.cascade_blocks:
            raise ValueError(f"cascade block must be 0..{self.cascade_blocks - 1}, got {block}")

        start = self.cascade_start_hours[block]
        end = start + self.cascade_horizon
        future_hours = f["future_hours"][start:end].astype(np.int16)
        if len(future_hours) != self.cascade_horizon:
            raise ValueError(f"{path}: block {block} has invalid future hour slice {start}:{end}")

        target_k = f[self.target_key][start:end, y_slice, x_slice].astype(np.float32)
        target_mask = f[self.target_valid_mask_key][start:end, y_slice, x_slice].astype(bool)
        target_mask = target_mask & np.isfinite(target_k)
        target = self.normalize_bt13(target_k)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        target = target[None, ...]
        target_k = np.nan_to_num(target_k, nan=np.nan, posinf=np.nan, neginf=np.nan)[None, ...]
        valid_mask = target_mask.astype(np.float32)[None, ...]
        loss_mask = valid_mask
        if self.loss_min_lon is not None:
            loss_lon = self._read_lon_grid(f, y_slice, x_slice)
            if loss_lon is None:
                raise ValueError("dataset.loss_min_lon requires a longitude grid")
            loss_domain = np.isfinite(loss_lon) & (loss_lon >= self.loss_min_lon)
            loss_mask = valid_mask * loss_domain[None, None, :, :]
        boundary_prev_k, boundary_prev_mask = self._read_boundary_previous(f, block, start, y_slice, x_slice)
        boundary_prev = self.normalize_bt13(boundary_prev_k)[None, None, ...]
        boundary_prev = np.nan_to_num(boundary_prev, nan=0.0, posinf=0.0, neginf=0.0)
        boundary_valid_mask = (boundary_prev_mask & target_mask[0]).astype(np.float32)[None, None, ...]
        west_edge_weight = self._west_edge_weight_mask(f, y_slice, x_slice)
        west_edge_weight = np.repeat(west_edge_weight[None, None, :, :], self.cascade_horizon, axis=1)

        history_k, history_mask, history_source = self._build_rolling_history(
            f=f,
            path=path,
            block=block,
            y_slice=y_slice,
            x_slice=x_slice,
            history_override=history_override,
            history_mask_override=history_mask_override,
        )
        condition, condition_names, condition_group_ids = self._build_condition(
            f=f,
            block=block,
            history_k=history_k,
            history_mask=history_mask,
            y_slice=y_slice,
            x_slice=x_slice,
        )
        wrf_cloud_condition, wrf_cloud_names = self._build_wrf_cloud_condition(
            f=f,
            start=start,
            end=end,
            y_slice=y_slice,
            x_slice=x_slice,
        )
        self.last_wrf_cloud_condition_names = wrf_cloud_names
        self.last_condition_names = condition_names
        aerosol_3d_condition = self._build_aerosol_3d_condition(
            f=f,
            start=start,
            end=end,
            y_slice=y_slice,
            x_slice=x_slice,
        )

        if FORBIDDEN_CONDITION_KEYS.intersection(condition_names):
            raise ValueError(f"Forbidden future-label condition keys present: {FORBIDDEN_CONDITION_KEYS.intersection(condition_names)}")

        spatial_valid = self._spatial_valid_mask(f, y_slice, x_slice)
        sample = {
            "condition": torch.from_numpy(np.ascontiguousarray(condition, dtype=np.float32)),
            "condition_group_ids": torch.from_numpy(
                np.ascontiguousarray(condition_group_ids, dtype=np.int64)
            ),
            "target": torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
            "target_k": torch.from_numpy(np.ascontiguousarray(target_k, dtype=np.float32)),
            "target_valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=np.float32)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask, dtype=np.float32)),
            "loss_mask": torch.from_numpy(np.ascontiguousarray(loss_mask, dtype=np.float32)),
            "boundary_prev_target": torch.from_numpy(np.ascontiguousarray(boundary_prev, dtype=np.float32)),
            "boundary_valid_mask": torch.from_numpy(np.ascontiguousarray(boundary_valid_mask, dtype=np.float32)),
            "west_edge_weight_mask": torch.from_numpy(np.ascontiguousarray(west_edge_weight, dtype=np.float32)),
            "bt13_history_k": torch.from_numpy(np.ascontiguousarray(history_k, dtype=np.float32)),
            "bt13_history_valid_mask": torch.from_numpy(np.ascontiguousarray(history_mask.astype(np.float32), dtype=np.float32)),
            "spatial_valid_mask": torch.from_numpy(np.ascontiguousarray(spatial_valid.astype(np.float32), dtype=np.float32)),
            "coszen_future": torch.from_numpy(np.ascontiguousarray(self._read_future_scalar(f, "coszen_future", start, end, y_slice, x_slice), dtype=np.float32)),
            "daylight_mask_future": torch.from_numpy(np.ascontiguousarray(self._read_future_mask(f, "daylight_mask_future", start, end, y_slice, x_slice), dtype=np.float32)),
            "future_hours": torch.from_numpy(np.ascontiguousarray(future_hours, dtype=np.int16)),
            "cascade_block": torch.tensor(block, dtype=torch.long),
            "block_start": torch.tensor(start, dtype=torch.long),
            "history_source": torch.tensor(history_source, dtype=torch.long),
            "y0": torch.tensor(0 if y_slice.start is None else y_slice.start, dtype=torch.long),
            "x0": torch.tensor(0 if x_slice.start is None else x_slice.start, dtype=torch.long),
            "path": str(path),
        }
        if wrf_cloud_condition is not None:
            sample["wrf_cloud_condition"] = torch.from_numpy(
                np.ascontiguousarray(wrf_cloud_condition, dtype=np.float32)
            )
        if aerosol_3d_condition is not None:
            sample["aerosol_3d_condition"] = torch.from_numpy(
                np.ascontiguousarray(aerosol_3d_condition, dtype=np.float32)
            )
        return sample

    def _validate_required_keys(self, path: Path, f: h5py.File) -> None:
        required = [
            self.target_key,
            self.target_valid_mask_key,
            self.history_key,
            self.history_valid_mask_key,
            "future_hours",
            "cloud_history_offsets_hours",
        ]
        for key in required:
            if key not in f:
                raise KeyError(f"{path}: missing required key {key}")
        future_hours = f["future_hours"][...].astype(int).tolist()
        if future_hours != list(range(1, self.forecast_hours + 1)):
            raise ValueError(f"{path}: future_hours must be 1..{self.forecast_hours}, got {future_hours}")
        history_offsets = f["cloud_history_offsets_hours"][...].astype(int).tolist()
        expected_offsets = list(range(-self.history_length + 1, 1))
        if history_offsets != expected_offsets:
            raise ValueError(f"{path}: cloud_history_offsets_hours must be {expected_offsets}, got {history_offsets}")
        if self.require_wrf_cloud:
            if "wrf_cloud_future" not in f or "wrf_cloud_var_names" not in f:
                raise KeyError(f"{path}: wrfcloud-v2 requires wrf_cloud_future and wrf_cloud_var_names")
            shape = tuple(f["wrf_cloud_future"].shape)
            expected_shape = (
                self.forecast_hours,
                *self.wrf_cloud_expected_spatial_shape,
                len(self.wrf_cloud_expected_names),
            )
            if shape != expected_shape:
                raise ValueError(f"{path}: wrf_cloud_future shape {shape} != expected {expected_shape}")
            target_shape = tuple(f[self.target_key].shape)
            expected_target_shape = (self.forecast_hours, *self.wrf_cloud_expected_spatial_shape)
            if target_shape != expected_target_shape:
                raise ValueError(f"{path}: {self.target_key} shape {target_shape} != expected {expected_target_shape}")
            names = self._read_names(f, "wrf_cloud_var_names")
            if names != self.wrf_cloud_expected_names:
                raise ValueError(
                    f"{path}: wrf_cloud_var_names order mismatch; got {names}, expected {self.wrf_cloud_expected_names}"
                )

    def _build_rolling_history(
        self,
        f: h5py.File,
        path: Path,
        block: int,
        y_slice: slice,
        x_slice: slice,
        history_override: np.ndarray | None,
        history_mask_override: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if history_override is not None:
            history_k = np.asarray(history_override, dtype=np.float32)
            if history_k.shape[0] != self.history_length:
                raise ValueError(f"history_override must have {self.history_length} frames, got {history_k.shape}")
            history_mask = np.isfinite(history_k)
            if history_mask_override is not None:
                history_mask = np.asarray(history_mask_override).astype(bool) & history_mask
            return history_k, history_mask, 2

        hist = f[self.history_key][:, y_slice, x_slice].astype(np.float32)
        hist_mask = f[self.history_valid_mask_key][:, y_slice, x_slice].astype(bool) & np.isfinite(hist)
        if block == 0:
            return hist, hist_mask, 0
        if not self.teacher_forcing:
            raise ValueError("teacher_forcing=false requires history_override for cascade blocks > 0")
        future_needed = self.cascade_start_hours[block]
        hist_tail = hist[future_needed:]
        hist_tail_mask = hist_mask[future_needed:]
        use_predicted = np.random.random() < self.predicted_history_probability()
        if use_predicted:
            future, future_mask = self._read_predicted_history_cache(path, future_needed, y_slice, x_slice)
            history_source = 1
        else:
            future = f[self.target_key][0:future_needed, y_slice, x_slice].astype(np.float32)
            future_mask = f[self.target_valid_mask_key][0:future_needed, y_slice, x_slice].astype(bool) & np.isfinite(future)
            future, future_mask = self._perturb_teacher_forced_future(future, future_mask, block=block)
            history_source = 0
        history_k = np.concatenate([hist_tail, future], axis=0)
        history_mask = np.concatenate([hist_tail_mask, future_mask], axis=0)
        if history_k.shape[0] != self.history_length:
            raise ValueError(f"Built rolling history with {history_k.shape[0]} frames, expected {self.history_length}")
        return history_k, history_mask, history_source

    def _perturb_teacher_forced_future(self, future: np.ndarray, future_mask: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
        if self.split != "train" or future.size == 0:
            return future, future_mask
        scale = self._teacher_perturb_scale(block)
        if scale <= 0.0:
            return future, future_mask
        future = np.asarray(future, dtype=np.float32).copy()
        future_mask = np.asarray(future_mask).astype(bool).copy()
        if self.history_teacher_noise_std_k > 0.0:
            noise = np.random.normal(0.0, self.history_teacher_noise_std_k * scale, size=future.shape).astype(np.float32)
            future = np.where(future_mask, future + noise, future)
        if self.history_teacher_bias_std_k > 0.0:
            bias = np.random.normal(0.0, self.history_teacher_bias_std_k * scale, size=(future.shape[0], 1, 1)).astype(np.float32)
            future = np.where(future_mask, future + bias, future)
        blur_prob = min(1.0, self.history_teacher_blur_prob * scale)
        if blur_prob > 0.0:
            blur_frames = np.random.random(size=(future.shape[0],)) < blur_prob
            if np.any(blur_frames):
                blurred = self._box_blur_valid(future, future_mask)
                future = np.where(blur_frames[:, None, None] & future_mask, blurred, future)
        if self.history_teacher_drop_prob > 0.0:
            drop_prob = min(0.75, self.history_teacher_drop_prob * scale)
            drop = (np.random.random(size=future_mask.shape) < drop_prob) & future_mask
            future_mask = future_mask & ~drop
            future = np.where(drop, np.nan, future)
        finite_mask = future_mask & np.isfinite(future)
        future = np.where(finite_mask, np.clip(future, self.b13_clip_min, self.b13_clip_max), future)
        return future, future_mask

    def _teacher_perturb_scale(self, block: int) -> float:
        if not self.history_teacher_block_scale:
            return 1.0
        idx = min(max(int(block), 0), len(self.history_teacher_block_scale) - 1)
        return float(self.history_teacher_block_scale[idx])

    @staticmethod
    def _box_blur_valid(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        valid = np.asarray(valid_mask).astype(bool)
        weighted = np.where(valid, values, 0.0)
        padded_values = np.pad(weighted, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
        padded_valid = np.pad(valid.astype(np.float32), ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
        total = np.zeros_like(values, dtype=np.float32)
        count = np.zeros_like(values, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                total += padded_values[:, dy : dy + values.shape[1], dx : dx + values.shape[2]]
                count += padded_valid[:, dy : dy + values.shape[1], dx : dx + values.shape[2]]
        blurred = total / np.maximum(count, 1.0)
        return np.where(valid, blurred, values)

    def _build_condition(
        self,
        f: h5py.File,
        block: int,
        history_k: np.ndarray,
        history_mask: np.ndarray,
        y_slice: slice,
        x_slice: slice,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        start = self.cascade_start_hours[block]
        end = start + self.cascade_horizon
        h, w = int(history_k.shape[1]), int(history_k.shape[2])
        parts: list[np.ndarray] = []
        names: list[str] = []
        group_ids: list[int] = []
        absolute_parts: list[np.ndarray] = []
        absolute_names: list[str] = []
        absolute_group_ids: list[int] = []

        def add_part(
            name: str,
            arr: np.ndarray,
            channel_names: list[str] | None = None,
            group: str = "wrf_regular",
        ) -> None:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError(f"Condition part {name} must be [C,T,H,W], got {arr.shape}")
            parts.append(arr)
            if channel_names is None:
                names.extend([name] * int(arr.shape[0]))
            else:
                names.extend(channel_names)
            group_ids.extend([CONDITION_GROUP_IDS[group]] * int(arr.shape[0]))

        if self.include_bt13_hist:
            hist_norm = self.normalize_bt13(history_k)
            hist_flat = hist_norm.reshape(self.history_length, h, w)
            hist_repeated = np.repeat(hist_flat[:, None, :, :], self.cascade_horizon, axis=1)
            add_part("bt13_hist_rolling", hist_repeated, [f"bt13_hist_t{i:02d}" for i in range(self.history_length)], group="history_b13")
            hist_mask_flat = history_mask.astype(np.float32).reshape(self.history_length, h, w)
            hist_mask_rep = np.repeat(hist_mask_flat[:, None, :, :], self.cascade_horizon, axis=1)
            add_part("bt13_hist_valid_mask_rolling", hist_mask_rep, [f"bt13_hist_valid_t{i:02d}" for i in range(self.history_length)], group="history_b13")

        if self.include_cloud_hist and "cloud_hist_feat" in f:
            cloud_hist = f["cloud_hist_feat"][:, y_slice, x_slice, :].astype(np.float32)
            cloud_names = self._read_names(f, "cloud_feature_names")
            cloud_channels = []
            cloud_channel_names = []
            for c in range(cloud_hist.shape[-1]):
                values = cloud_hist[..., c]
                channel_name = cloud_names[c] if c < len(cloud_names) else f"cloud_hist_{c}"
                if channel_name == "CLOT":
                    values = normalize_log_cot(log1p_cot(values, self.cot_clip_max), self.cot_clip_max)
                else:
                    values = clean_finite(values)
                cloud_channels.append(values)
                cloud_channel_names.extend([f"cloud_hist_{channel_name}_t{i:02d}" for i in range(values.shape[0])])
            if cloud_channels:
                cloud_arr = np.stack(cloud_channels, axis=0).reshape(-1, h, w)
                cloud_arr = np.repeat(cloud_arr[:, None, :, :], self.cascade_horizon, axis=1)
                add_part("cloud_hist_feat", cloud_arr, cloud_channel_names, group="history_b13")
            if "cloud_hist_valid_mask" in f:
                cloud_mask = f["cloud_hist_valid_mask"][:, y_slice, x_slice].astype(np.float32)
                cloud_mask = np.repeat(cloud_mask[:, None, :, :], self.cascade_horizon, axis=1)
                add_part("cloud_hist_valid_mask", cloud_mask, [f"cloud_hist_valid_t{i:02d}" for i in range(cloud_mask.shape[0])], group="history_b13")

        if self.include_coszen:
            if "coszen_hist" in f:
                arr = f["coszen_hist"][:, y_slice, x_slice].astype(np.float32)
                arr = np.repeat(arr[:, None, :, :], self.cascade_horizon, axis=1)
                add_part("coszen_hist", arr, [f"coszen_hist_t{i:02d}" for i in range(arr.shape[0])], group="static_time")
            if "coszen_future" in f:
                arr = f["coszen_future"][start:end, y_slice, x_slice].astype(np.float32)[None, ...]
                add_part("coszen_future", arr, group="static_time")

        if self.include_daylight_mask:
            if "daylight_mask_hist" in f:
                arr = f["daylight_mask_hist"][:, y_slice, x_slice].astype(np.float32)
                arr = np.repeat(arr[:, None, :, :], self.cascade_horizon, axis=1)
                add_part("daylight_mask_hist", arr, [f"daylight_hist_t{i:02d}" for i in range(arr.shape[0])], group="static_time")
            if "daylight_mask_future" in f:
                arr = f["daylight_mask_future"][start:end, y_slice, x_slice].astype(np.float32)[None, ...]
                add_part("daylight_mask_future", arr, group="static_time")

        if self.include_wrf_meteo and "wrf_meteo_future" in f:
            arr = f["wrf_meteo_future"][start:end, y_slice, x_slice, :].astype(np.float32)
            var_names = self._read_names(f, "meteo_var_names")
            add_part("wrf_meteo_future", np.moveaxis(arr, -1, 0), [f"wrf_meteo_{n}" for n in var_names] or None)
        if self.include_wrf_rad and "wrf_rad_future" in f:
            arr = f["wrf_rad_future"][start:end, y_slice, x_slice, :].astype(np.float32)
            var_names = self._read_names(f, "rad_var_names")
            add_part("wrf_rad_future", np.moveaxis(arr, -1, 0), [f"wrf_rad_{n}" for n in var_names] or None)
        if self.include_aod550:
            if self.aod550_key not in f:
                if self.require_aod550:
                    raise KeyError(f"Required aerosol optical depth field is missing: {self.aod550_key}")
            else:
                add_part(
                    self.aod550_key,
                    f[self.aod550_key][start:end, y_slice, x_slice].astype(np.float32)[None, ...],
                )
        if self.include_k_wrf and "k_wrf_future" in f:
            add_part("k_wrf_future", f["k_wrf_future"][start:end, y_slice, x_slice].astype(np.float32)[None, ...])
            if "k_wrf_valid_mask" in f:
                add_part("k_wrf_valid_mask", f["k_wrf_valid_mask"][start:end, y_slice, x_slice].astype(np.float32)[None, ...])
        if self.include_aerosol_column_summary:
            if "aerosol_column_summary_future" not in f:
                if self.require_aerosol_column_summary:
                    raise KeyError("Required aerosol_column_summary_future field is missing")
            else:
                var_names = self._read_names(f, "aerosol_column_summary_var_names")
                if self.aerosol_column_summary_selected_names is not None:
                    if len(var_names) != len(set(var_names)):
                        raise ValueError("aerosol_column_summary_var_names contains duplicate names")
                    missing = [
                        name for name in self.aerosol_column_summary_selected_names if name not in var_names
                    ]
                    if missing:
                        raise ValueError(f"Requested aerosol column summaries are missing: {missing}")
                    indices = [var_names.index(name) for name in self.aerosol_column_summary_selected_names]
                    arr = f["aerosol_column_summary_future"][
                        start:end, y_slice, x_slice, indices
                    ].astype(np.float32)
                    var_names = list(self.aerosol_column_summary_selected_names)
                else:
                    arr = f["aerosol_column_summary_future"][
                        start:end, y_slice, x_slice, :
                    ].astype(np.float32)
                add_part(
                    "aerosol_column_summary_future",
                    np.moveaxis(arr, -1, 0),
                    [f"aerosol_summary_{n}" for n in var_names] or None,
                )
        # The vertical encoder receives [species, level, T, H, W] separately
        # from the regular condition tensor.
        if self.include_static and "static_feat" in f:
            static = f["static_feat"][y_slice, x_slice, :].astype(np.float32)
            static = np.moveaxis(static, -1, 0)
            static = np.repeat(static[:, None, :, :], self.cascade_horizon, axis=1)
            var_names = self._read_names(f, "static_var_names")
            add_part("static_feat", static, [f"static_{n}" for n in var_names] or None, group="static_time")
        if self.include_time:
            time_parts = []
            time_names = []
            for key in ["lead_time_norm", "hour_utc_sin", "hour_utc_cos", "hour_bjt_sin", "hour_bjt_cos"]:
                if key in f:
                    values = f[key][start:end].astype(np.float32)
                elif key == "lead_time_norm":
                    values = f["future_hours"][start:end].astype(np.float32) / 24.0
                else:
                    values = np.zeros(self.cascade_horizon, dtype=np.float32)
                time_parts.append(np.repeat(values[None, :, None, None], h, axis=2).repeat(w, axis=3))
                time_names.append(key)
            add_part("future_time_encoding", np.concatenate(time_parts, axis=0), time_names, group="static_time")

        if not parts:
            raise ValueError("No condition channels were selected")
        condition = np.concatenate(parts, axis=0)
        condition = channelwise_standardize(condition)

        # Absolute anchors must be appended *after* sample-wise condition
        # standardization.  The history anchor intentionally consumes the
        # already assembled rolling history (teacher, cache, or autoregressive
        # override), never bt13_future directly.
        if self.include_bt13_hist_absolute:
            hist_absolute = self.normalize_bt13(history_k).reshape(self.history_length, h, w)
            hist_absolute = np.repeat(
                hist_absolute[:, None, :, :], self.cascade_horizon, axis=1
            )
            absolute_parts.append(hist_absolute.astype(np.float32, copy=False))
            absolute_names.extend(
                [f"bt13_hist_absolute_t{i:02d}" for i in range(self.history_length)]
            )
            absolute_group_ids.extend(
                [CONDITION_GROUP_IDS["history_b13"]] * self.history_length
            )

        if self.include_absolute_lead_time:
            lead_values = (
                f["future_hours"][start:end].astype(np.float32)
                / float(self.forecast_hours)
            )
            absolute_lead = np.broadcast_to(
                lead_values[None, :, None, None],
                (1, self.cascade_horizon, h, w),
            ).copy()
            absolute_parts.append(absolute_lead)
            absolute_names.append("absolute_lead_time_norm")
            absolute_group_ids.append(CONDITION_GROUP_IDS["static_time"])

        if absolute_parts:
            condition = np.concatenate([condition, *absolute_parts], axis=0)
            names.extend(absolute_names)
            group_ids.extend(absolute_group_ids)

        return condition, names, np.asarray(group_ids, dtype=np.int64)

    def _build_aerosol_3d_condition(
        self,
        *,
        f: h5py.File,
        start: int,
        end: int,
        y_slice: slice,
        x_slice: slice,
    ) -> np.ndarray | None:
        """Read and standardize full WRF-Chem aerosol profiles.

        The source layout is [T, Z, H, W, species].  Per species/layer
        standardization uses only the selected forecast window and spatial
        patch, retaining vertical contrasts while AOD/column-summary inputs
        retain the absolute column-scale information.
        """

        if not self.include_aerosol_full3d:
            return None
        if self.aerosol_full3d_cache_dir is not None:
            cache_path = self._aerosol_full3d_cache_path(Path(f.filename))
            if cache_path.is_file():
                block = self.cascade_start_hours.index(int(start))
                return self._read_aerosol_full3d_cache(Path(f.filename), block, y_slice, x_slice)
            if self.require_aerosol_full3d_cache:
                raise RuntimeError(f"Required 3-D aerosol cache is missing: {cache_path}")
        if "aerosol_future" not in f:
            raise KeyError("include_aerosol_full3d=true requires aerosol_future")
        raw = f["aerosol_future"][start:end, :, y_slice, x_slice, :].astype(np.float32)
        if raw.ndim != 5:
            raise ValueError(f"aerosol_future must be [T,Z,H,W,species], got {raw.shape}")
        expected = (
            self.cascade_horizon,
            self.aerosol_full3d_expected_levels,
            raw.shape[2],
            raw.shape[3],
            self.aerosol_full3d_expected_species,
        )
        if raw.shape != expected:
            raise ValueError(
                "aerosol_future shape mismatch: "
                f"got {raw.shape}, expected {expected} for the selected patch"
            )
        # [T,Z,H,W,A] -> [A,Z,T,H,W]; treat each species/layer as a channel
        # for normalization, then restore the explicit vertical axis.
        arranged = np.moveaxis(raw, -1, 0).transpose(0, 2, 1, 3, 4)
        flat = arranged.reshape(arranged.shape[0] * arranged.shape[1], -1)
        finite = np.isfinite(flat)
        means = np.zeros((flat.shape[0], 1), dtype=np.float32)
        stds = np.ones((flat.shape[0], 1), dtype=np.float32)
        for channel in range(flat.shape[0]):
            values = flat[channel, finite[channel]]
            if values.size:
                means[channel, 0] = values.mean(dtype=np.float64)
                std = values.std(dtype=np.float64)
                stds[channel, 0] = std if std > 1.0e-6 else 1.0
        standardized = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        standardized = np.clip((standardized - means) / stds, -10.0, 10.0)
        return standardized.reshape(arranged.shape).astype(np.float32, copy=False)

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

    def _spatial_valid_mask(self, f: h5py.File, y_slice: slice, x_slice: slice) -> np.ndarray:
        if "himawari_on_wrf_valid_mask" in f:
            return f["himawari_on_wrf_valid_mask"][y_slice, x_slice].astype(bool)
        hist_mask = f[self.history_valid_mask_key][:, y_slice, x_slice].astype(bool)
        return hist_mask.any(axis=0)

    def _read_boundary_previous(self, f: h5py.File, block: int, start: int, y_slice: slice, x_slice: slice) -> tuple[np.ndarray, np.ndarray]:
        if block <= 0:
            prev_k = f[self.history_key][-1, y_slice, x_slice].astype(np.float32)
            prev_mask = f[self.history_valid_mask_key][-1, y_slice, x_slice].astype(bool)
        else:
            prev_idx = max(0, int(start) - 1)
            prev_k = f[self.target_key][prev_idx, y_slice, x_slice].astype(np.float32)
            prev_mask = f[self.target_valid_mask_key][prev_idx, y_slice, x_slice].astype(bool)
        prev_mask = prev_mask & np.isfinite(prev_k)
        return prev_k, prev_mask

    def _west_edge_weight_mask(self, f: h5py.File, y_slice: slice, x_slice: slice) -> np.ndarray:
        target_shape = f[self.target_key].shape[1:3]
        if y_slice.start is None and y_slice.stop is None:
            h = int(target_shape[0])
        else:
            h = int((y_slice.stop or target_shape[0]) - (y_slice.start or 0))
        if x_slice.start is None and x_slice.stop is None:
            w = int(target_shape[1])
        else:
            w = int((x_slice.stop or target_shape[1]) - (x_slice.start or 0))
        if not self.west_edge_weighting:
            return np.ones((h, w), dtype=np.float32)
        lon = self._read_lon_grid(f, y_slice, x_slice)
        if lon is None:
            return np.ones((h, w), dtype=np.float32)
        weight = np.ones(lon.shape, dtype=np.float32)
        weight = np.where(lon < self.west_edge_transition_lon, self.west_edge_transition_weight, weight)
        weight = np.where(lon < self.west_edge_lon_threshold, self.west_edge_weight, weight)
        return np.where(np.isfinite(lon), weight, 1.0).astype(np.float32)

    def _read_lon_grid(self, f: h5py.File, y_slice: slice, x_slice: slice) -> np.ndarray | None:
        if "grid_lon" in f:
            return f["grid_lon"][y_slice, x_slice].astype(np.float32)
        if "static_feat" in f:
            names = self._read_names(f, "static_var_names")
            for candidate in ["XLONG", "lon", "longitude", "LON"]:
                if candidate in names:
                    idx = names.index(candidate)
                    return f["static_feat"][y_slice, x_slice, idx].astype(np.float32)
        return None

    def _read_future_scalar(self, f: h5py.File, key: str, start: int, end: int, y_slice: slice, x_slice: slice) -> np.ndarray:
        if key in f:
            return f[key][start:end, y_slice, x_slice].astype(np.float32)
        shape = (self.cascade_horizon, y_slice.stop - (y_slice.start or 0), x_slice.stop - (x_slice.start or 0))
        return np.zeros(shape, dtype=np.float32)

    def _read_future_mask(self, f: h5py.File, key: str, start: int, end: int, y_slice: slice, x_slice: slice) -> np.ndarray:
        if key in f:
            return f[key][start:end, y_slice, x_slice].astype(np.float32)
        return np.zeros((self.cascade_horizon, y_slice.stop - (y_slice.start or 0), x_slice.stop - (x_slice.start or 0)), dtype=np.float32)

    @staticmethod
    def _read_names(f: h5py.File, key: str) -> list[str]:
        if key not in f:
            return []
        out = []
        for value in f[key][()]:
            if isinstance(value, (bytes, bytearray, np.bytes_)):
                out.append(value.decode("utf-8", errors="replace"))
            else:
                out.append(str(value))
        return out

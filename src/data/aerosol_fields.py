from __future__ import annotations

from typing import Sequence

import numpy as np


NATIVE_AEROSOL_NAMES = [
    "so4",
    "BC1",
    "BC2",
    "OC1",
    "OC2",
    "DUST_1",
    "DUST_2",
    "DUST_3",
    "DUST_4",
    "DUST_5",
    "SEAS_1",
    "SEAS_2",
    "SEAS_3",
    "SEAS_4",
]

AEROSOL_CHANNEL_NAMES = [
    "SO4",
    "BC1",
    "BC2",
    "OC1",
    "OC2",
    "DUST_fine",
    "DUST_coarse",
    "SEAS_fine",
    "SEAS_coarse",
    "extinction550_kohler",
]

AEROSOL_SUMMARY_NAMES = [
    *(f"{name}_column_mean" for name in NATIVE_AEROSOL_NAMES),
    *(f"{name}_column_sum" for name in NATIVE_AEROSOL_NAMES),
]

AGL_EDGES_M = np.arange(0.0, 16000.0, 1000.0, dtype=np.float32)
# Numerical interpolation can leave physically-zero tracer cells at about
# 1e-23 ug kg-1.  Preserve a hard failure for meaningful negatives while
# clipping only this floating-point residue before physical-height remapping.
NUMERICAL_NEGATIVE_TOLERANCE_UGKG = np.float32(1.0e-12)
NUMERICAL_NEGATIVE_TOLERANCE_KM1 = np.float32(1.0e-12)


def decode_strings(values: Sequence[object]) -> list[str]:
    return [
        value.decode("utf-8", errors="strict")
        if isinstance(value, (bytes, bytearray, np.bytes_))
        else str(value)
        for value in values
    ]


def _to_ug_per_kg(values: np.ndarray, unit: str, name: str) -> np.ndarray:
    normalized = unit.lower().replace("µ", "u").replace("μ", "u").strip()
    if normalized.startswith("ug kg-1"):
        return values.astype(np.float32, copy=False)
    if normalized.startswith("g kg-1"):
        return (values * np.float32(1.0e6)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported aerosol unit for {name}: {unit!r}")


def composition_9_from_native(
    aerosol: np.ndarray,
    names: Sequence[str],
    units: Sequence[str],
) -> tuple[np.ndarray, dict[str, float]]:
    """Return nine composition channels in ug kg-1 from [..., species]."""

    aerosol = np.asarray(aerosol, dtype=np.float32)
    names = [str(name) for name in names]
    units = [str(unit) for unit in units]
    if aerosol.ndim < 2 or aerosol.shape[-1] != len(names) or len(names) != len(units):
        raise ValueError(
            f"Aerosol/name/unit dimensions disagree: {aerosol.shape}, {len(names)}, {len(units)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("aerosol_var_names contains duplicate entries")
    missing = [name for name in NATIVE_AEROSOL_NAMES if name not in names]
    if missing:
        raise ValueError(f"Missing native aerosol fields: {missing}")
    native = {
        name: _to_ug_per_kg(aerosol[..., names.index(name)], units[names.index(name)], name)
        for name in NATIVE_AEROSOL_NAMES
    }
    derived = {
        "DUST_fine": native["DUST_1"] + native["DUST_2"],
        "DUST_coarse": native["DUST_3"] + native["DUST_4"] + native["DUST_5"],
        "SEAS_fine": native["SEAS_1"] + native["SEAS_2"],
        "SEAS_coarse": native["SEAS_3"] + native["SEAS_4"],
    }
    closure: dict[str, float] = {}
    for name, expected in derived.items():
        if name in names:
            observed = _to_ug_per_kg(
                aerosol[..., names.index(name)], units[names.index(name)], name
            )
            closure[name] = float(np.max(np.abs(observed - expected)))
    channels = [
        native["so4"],
        native["BC1"],
        native["BC2"],
        native["OC1"],
        native["OC2"],
        derived["DUST_fine"],
        derived["DUST_coarse"],
        derived["SEAS_fine"],
        derived["SEAS_coarse"],
    ]
    return np.stack(channels, axis=0).astype(np.float32, copy=False), closure


def remap_profiles_to_agl(
    composition_9: np.ndarray,
    extinction_km1: np.ndarray,
    z_interface_agl_m: np.ndarray,
    edges_m: np.ndarray = AGL_EDGES_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    """Map eta layers to fixed AGL layers using geometric overlap.

    Composition is averaged over the covered part of each target layer.
    Extinction is divided by the fixed target-layer thickness, so integrating
    the remapped field exactly recovers source optical depth within the range.
    """

    q = np.asarray(composition_9, dtype=np.float32)
    extinction = np.asarray(extinction_km1, dtype=np.float32)
    z = np.maximum(np.asarray(z_interface_agl_m, dtype=np.float32), 0.0)
    edges = np.asarray(edges_m, dtype=np.float32)
    if q.ndim != 4 or extinction.ndim != 3 or z.ndim != 3:
        raise ValueError(f"Expected q[C,Z,H,W], extinction[Z,H,W], z[Z+1,H,W], got {q.shape}, {extinction.shape}, {z.shape}")
    if q.shape[1:] != extinction.shape or z.shape != (extinction.shape[0] + 1, *extinction.shape[1:]):
        raise ValueError("Composition, extinction, and height grids do not align")
    dz = z[1:] - z[:-1]
    if np.any(dz <= 0.0):
        raise ValueError(f"z_interface_agl is not strictly increasing; violations={int(np.count_nonzero(dz <= 0.0))}")
    if not np.isfinite(q).all() or not np.isfinite(extinction).all() or not np.isfinite(z).all():
        raise ValueError("Non-finite values found in 3-D aerosol inputs")
    q_negative = q < 0.0
    extinction_negative = extinction < 0.0
    q_minimum = float(q.min())
    extinction_minimum = float(extinction.min())
    if np.any(q < -NUMERICAL_NEGATIVE_TOLERANCE_UGKG) or np.any(
        extinction < -NUMERICAL_NEGATIVE_TOLERANCE_KM1
    ):
        raise ValueError(
            "Meaningful negative values found in 3-D aerosol inputs: "
            f"q_min={q_minimum:.6g}, extinction_min={extinction_minimum:.6g}"
        )
    # Do not mutate arrays that originated in an HDF5 slice.  The cache
    # manifest records the corresponding QC counts for traceability.
    q = np.maximum(q, 0.0)
    extinction = np.maximum(extinction, 0.0)

    levels = len(edges) - 1
    q_out = np.zeros((q.shape[0], levels, *q.shape[2:]), dtype=np.float32)
    extinction_out = np.zeros((levels, *extinction.shape[1:]), dtype=np.float32)
    coverage = np.zeros_like(extinction_out)
    lower = z[:-1]
    upper = z[1:]
    for level in range(levels):
        lo = edges[level]
        hi = edges[level + 1]
        overlap_m = np.maximum(np.minimum(upper, hi) - np.maximum(lower, lo), 0.0)
        covered_m = overlap_m.sum(axis=0, dtype=np.float32)
        coverage[level] = np.clip(covered_m / (hi - lo), 0.0, 1.0)
        weighted_q = np.einsum("czhw,zhw->chw", q, overlap_m, optimize=True)
        q_out[:, level] = np.divide(
            weighted_q,
            covered_m[None, ...],
            out=np.zeros_like(weighted_q),
            where=covered_m[None, ...] > 0.0,
        )
        extinction_out[level] = (
            np.sum(extinction * overlap_m, axis=0, dtype=np.float32) / np.float32(hi - lo)
        )

    source_aod = np.sum(extinction * dz / np.float32(1000.0), axis=0, dtype=np.float32)
    below_aod = np.sum(
        extinction_out * (np.diff(edges)[:, None, None] / np.float32(1000.0)),
        axis=0,
        dtype=np.float32,
    )
    qc: dict[str, np.ndarray | float] = {
        "coverage": coverage,
        "source_aod": source_aod,
        "aod_0_15km": below_aod,
        "aod_above_15km": np.maximum(source_aod - below_aod, 0.0),
        "minimum_coverage": float(coverage.min()),
        "input_aerosol_negative_clamped_count": int(np.count_nonzero(q_negative)),
        "input_extinction_negative_clamped_count": int(np.count_nonzero(extinction_negative)),
        "input_aerosol_min_before_clamp_ugkg": q_minimum,
        "input_extinction_min_before_clamp_km1": extinction_minimum,
    }
    return q_out, extinction_out, qc


def horizontal_mean_pool(values: np.ndarray, pool: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if pool < 1:
        raise ValueError("pool must be >= 1")
    if pool == 1:
        return values
    height, width = values.shape[-2:]
    pooled_h = (height + pool - 1) // pool
    pooled_w = (width + pool - 1) // pool
    padded = np.zeros((*values.shape[:-2], pooled_h * pool, pooled_w * pool), dtype=np.float32)
    padded[..., :height, :width] = values
    blocks = padded.reshape(*values.shape[:-2], pooled_h, pool, pooled_w, pool)
    pooled = blocks.sum(axis=(-3, -1), dtype=np.float32)
    row_counts = np.full(pooled_h, pool, dtype=np.float32)
    col_counts = np.full(pooled_w, pool, dtype=np.float32)
    row_counts[-1] = height - (pooled_h - 1) * pool
    col_counts[-1] = width - (pooled_w - 1) * pool
    return pooled / (row_counts[:, None] * col_counts[None, :])


def transform_aerosol_features(
    aerosol: np.ndarray,
    aerosol_names: Sequence[str],
    aerosol_units: Sequence[str],
    extinction_km1: np.ndarray,
    z_interface_agl_m: np.ndarray,
    *,
    pool: int = 2,
) -> tuple[np.ndarray, dict[str, np.ndarray | float | dict[str, float]]]:
    """Build log-transformed [10,15,H_pool,W_pool] features for one hour."""

    q9, aggregate_closure = composition_9_from_native(aerosol, aerosol_names, aerosol_units)
    q_agl, ext_agl, qc = remap_profiles_to_agl(q9, extinction_km1, z_interface_agl_m)
    q_pooled = horizontal_mean_pool(q_agl, pool=pool)
    ext_pooled = horizontal_mean_pool(ext_agl, pool=pool)
    transformed = np.concatenate(
        [
            np.log1p(np.maximum(q_pooled, 0.0) / np.float32(1.0)),
            np.log1p(np.maximum(ext_pooled, 0.0)[None, ...] / np.float32(0.01)),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    qc["aggregate_closure_max_abs_ugkg"] = aggregate_closure
    return transformed, qc

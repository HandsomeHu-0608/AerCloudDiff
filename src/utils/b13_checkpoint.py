from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def checkpoint_fingerprint(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _ema_state(checkpoint: dict[str, Any]):
    state = checkpoint.get("ema_model_state")
    if state is not None:
        return state
    ema_state = checkpoint.get("ema_state")
    if isinstance(ema_state, dict):
        return ema_state.get("shadow")
    legacy = checkpoint.get("ema")
    if isinstance(legacy, dict):
        return legacy
    return None


def _raw_state(checkpoint: dict[str, Any]):
    return checkpoint.get("raw_model_state", checkpoint.get("model_state", checkpoint.get("model_state_dict")))


def select_checkpoint_state(
    checkpoint: dict[str, Any],
    weights: str = "auto",
    *,
    allow_fallback: bool = True,
):
    selection = str(weights).lower()
    if selection not in {"auto", "ema", "raw"}:
        raise ValueError("weights must be auto, ema, or raw")
    ema = _ema_state(checkpoint)
    raw = _raw_state(checkpoint)
    if selection == "ema":
        if ema is not None:
            return ema, "ema"
        if allow_fallback and raw is not None:
            return raw, "raw_fallback"
        raise KeyError("Checkpoint has no EMA model state")
    if selection == "raw":
        if raw is not None:
            return raw, "raw"
        if allow_fallback and ema is not None:
            return ema, "ema_fallback"
        raise KeyError("Checkpoint has no raw model state")
    if ema is not None:
        return ema, "ema"
    if raw is not None:
        return raw, "raw"
    raise KeyError("Checkpoint has no supported model state")


def requested_weights(use_ema: bool | None, *, explicit: bool = True) -> tuple[str, bool]:
    if use_ema is None:
        return "auto", True
    return ("ema" if use_ema else "raw"), not explicit


def load_checkpoint_weights(
    model,
    checkpoint: dict[str, Any],
    *,
    weights: str = "auto",
    strict: bool = True,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    state, loaded = select_checkpoint_state(checkpoint, weights=weights, allow_fallback=allow_fallback)
    incompatible = model.load_state_dict(state, strict=strict)
    return {
        "weights_requested": weights,
        "weights_loaded": loaded,
        "missing_keys": list(getattr(incompatible, "missing_keys", [])),
        "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
    }


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temp.replace(path)

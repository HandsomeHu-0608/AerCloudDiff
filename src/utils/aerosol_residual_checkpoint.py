"""Strict one-time migration of a qualified 2-D backbone into a 3-D adapter."""
from __future__ import annotations

from src.utils.b13_checkpoint import select_checkpoint_state


ADAPTER_ONLY_CHECKPOINT_SCHEMA = "aerclouddiff_3daer_b13_checkpoint"
ADAPTER_ONLY_PREFLIGHT_SCHEMA = "aerclouddiff_3daer_adapter_preflight"


def load_aerosol_residual_backbone(model, checkpoint: dict, *, weights: str = "ema") -> dict:
    """Load only a *complete* matching 2-D state and zero the adapter output.

    The caller owns checkpoint path/fingerprint validation.  This function
    deliberately rejects missing EMA weights, old 3-D checkpoints, partial
    states and shape mismatches.  It never migrates optimizer/epoch state.
    Resume of an existing adapter run must use the ordinary full-state loader.
    """
    if not hasattr(model, "aerosol_residual_adapter") or not hasattr(model, "adapter_prefix"):
        raise TypeError("Backbone migration requires a grouped aerosol residual model")
    state, loaded = select_checkpoint_state(checkpoint, weights=weights, allow_fallback=False)
    if not isinstance(state, dict) or not state:
        raise ValueError("Baseline checkpoint state must be a nonempty tensor dictionary")
    # DDP prefix removal is safe only when every key has the same prefix.
    if all(name.startswith("module.") for name in state):
        state = {name[len("module."):]: value for name, value in state.items()}
    current = model.state_dict()
    backbone = {name: value for name, value in current.items() if not name.startswith(model.adapter_prefix)}
    missing = sorted(set(backbone) - set(state))
    unexpected = sorted(set(state) - set(backbone))
    if missing or unexpected:
        raise ValueError(f"2-D backbone state keys do not match: missing={missing}, unexpected={unexpected}")
    mismatched = [
        f"{name}: checkpoint={getattr(value, 'shape', None)}, model={tuple(backbone[name].shape)}"
        for name, value in state.items()
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(backbone[name].shape)
    ]
    if mismatched:
        raise ValueError("2-D backbone tensor shapes do not match: " + "; ".join(mismatched))
    current.update(state)
    model.load_state_dict(current, strict=True)
    model.aerosol_residual_adapter.zero_output_projection()
    reset_modulation = getattr(model.aerosol_residual_adapter, "reset_condition_modulation", None)
    if reset_modulation is not None:
        reset_modulation()
    return {
        "weights_requested": weights,
        "weights_loaded": loaded,
        "migration": getattr(model, "backbone_migration_id", "strict_2d_backbone_zero_residual_v1"),
        "backbone_tensor_count": len(backbone),
        "adapter_initialization": "zero_output_projection",
        "missing_keys": [],
        "unexpected_keys": [],
    }

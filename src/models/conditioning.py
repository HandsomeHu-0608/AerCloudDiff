from __future__ import annotations


def describe_conditioning(config: dict) -> str:
    dataset = config.get("dataset", {})
    mode = dataset.get("aerosol_condition_mode", "none")
    use_wrf = dataset.get("use_future_wrf_condition", "auto")
    return f"use_future_wrf_condition={use_wrf}, aerosol_condition_mode={mode}, include_so4_future={dataset.get('include_so4_future', False)}"


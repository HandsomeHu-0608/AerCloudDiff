"""Frame-wise metrics and date/lead summaries for training and checkpoint selection."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import date, timedelta
import csv
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np

from src.utils.b13_checkpoint import write_json_atomic

METRICS = (
    "rmse_k", "mae_k", "bias_k", "abs_bias_k", "pearson_r", "ms_ssim",
    "fss_240k_11px", "fss_230k_11px", "fss_220k_11px",
)
SELECTION_TOLERANCE_K = 1e-6
NORMALIZATION_SHA256 = "e366a92b44432fdb3839ab30566db1f573ebc5b1ac08465ac927af385753105d"
COMPATIBLE_NORMALIZATION_SHA256 = {
    NORMALIZATION_SHA256,
    "22a3c1c2efca664f10f42ffd12c8008df9aff87ca0f556bc154c54ea2b427c86",
}
BASELINE_SHA256 = "aaca05af4ba33af4590925ac8daab79c67212cccf713d851944ebc59f1627f7e"
SAMPLING = dict(sampler="ddim", steps=100, eta=0.0, guidance_scale=0.0,
                clamp_mode="per_step", clamp_min=0.0, clamp_max=1.0,
                use_ema=True, seed=42, seed_mode="sample_keyed", ensemble_members=1)


def expected_dates(split: str) -> list[str]:
    limits = {"train": (date(2025, 1, 1), date(2025, 10, 31)),
              "val": (date(2025, 11, 1), date(2025, 11, 30)),
              "test": (date(2025, 12, 1), date(2025, 12, 31))}
    if split not in limits:
        raise ValueError(f"Unsupported final-run split: {split}")
    first, last = limits[split]
    return [(first + timedelta(days=i)).strftime("%Y%m%d")
            for i in range((last - first).days + 1)]


def sample_date(sample: str) -> str:
    found = re.search(r"(?:^|_)(2025\d{4})12(?=_|\.|$)", Path(sample).name)
    if found is None:
        raise ValueError(f"Expected daily 12UTC sample name, got {sample!r}")
    return found.group(1)


def validate_dates(samples: list[str], split: str) -> None:
    actual = sorted(sample_date(name) for name in samples)
    if actual != expected_dates(split):
        raise ValueError(f"{split}: expected the complete unique date list; got {len(actual)} samples")


def prediction_rows(data: Any, sample: str, *, min_lon: float = 80.0,
                    expected_shape: tuple[int, int, int] | None = (24, 149, 191),
                    target_policy: dict | None = None) -> list[dict]:
    from scripts.cloud_metrics import fractions_skill_score
    from scripts.analyze_cloud_structure_metrics import ms_ssim
    from src.data.final_target_policy import missing_leads

    pred = np.asarray(data["pred_k"], dtype=np.float64)
    truth = np.asarray(data["truth_k"], dtype=np.float64)
    valid = np.asarray(data["valid"], dtype=bool)
    lon = np.asarray(data["lon"], dtype=np.float64)
    if pred.ndim != 3 or pred.shape != truth.shape or valid.shape != truth.shape:
        raise ValueError(f"{sample}: prediction/truth/mask shape mismatch")
    if expected_shape is not None and pred.shape != expected_shape:
        raise ValueError(f"{sample}: expected shape {expected_shape}, got {pred.shape}")
    if lon.shape != pred.shape[1:] or not np.isfinite(lon).all():
        raise ValueError(f"{sample}: missing, nonfinite or mismatched longitude grid")
    if not np.array_equal(data["future_hours"], np.arange(1, pred.shape[0] + 1)):
        raise ValueError(f"{sample}: invalid lead axis")
    domain = valid & (lon >= min_lon)[None]
    if not np.isfinite(truth[domain]).all() or not np.isfinite(pred[domain]).all():
        raise ValueError(f"{sample}: nonfinite truth/prediction in the fixed valid domain")
    rows = []
    declared_missing = set(missing_leads(target_policy, sample)) if target_policy is not None else set()
    for i in range(pred.shape[0]):
        mask = domain[i]
        declared_absent = i + 1 in declared_missing
        unobserved = target_policy is None and not valid[i].any()
        if declared_absent or unobserved:
            # Preserve a predicted frame with no valid verification observations.
            # Geographic cropping alone cannot qualify as an absent target.
            if declared_absent and (valid[i].any() or np.isfinite(truth[i]).any()):
                raise ValueError(f"{sample}: declared missing target changed at lead {i + 1}")
            prediction_domain = lon >= min_lon
            if "spatial_valid" in data:
                spatial_valid = np.asarray(data["spatial_valid"], dtype=bool)
                if spatial_valid.shape != lon.shape:
                    raise ValueError(f"{sample}: mismatched spatial prediction mask")
                prediction_domain = prediction_domain & spatial_valid
            if not prediction_domain.any() or not np.isfinite(pred[i][prediction_domain]).all():
                raise ValueError(f"{sample}: empty or nonfinite prediction domain at missing lead {i + 1}")
            rows.append({"sample": sample, "lead_hour": i + 1, "valid_count": 0,
                         "known_missing_target": declared_absent,
                         "status": "known_missing_target" if declared_absent else "no_valid_observations",
                         **{metric: float("nan") for metric in METRICS}})
            continue
        if not mask.any():
            raise ValueError(f"{sample}: empty valid domain at lead {i + 1}")
        p, t = pred[i][mask], truth[i][mask]
        error = p - t
        bias = float(error.mean())
        correlation = (float(np.corrcoef(p, t)[0, 1])
                       if len(p) > 1 and p.std() > 0 and t.std() > 0 else float("nan"))
        # The original masked MS-SSIM uses multiply-by-mask while downsampling.
        # Replacing invalid entries first avoids NaN*0 without changing support.
        structure = ms_ssim(np.where(mask, pred[i], 0.0), np.where(mask, truth[i], 0.0), mask)
        row = {"sample": sample, "lead_hour": i + 1, "valid_count": int(mask.sum()),
               "rmse_k": float(np.sqrt(np.mean(error * error))),
               "mae_k": float(np.mean(np.abs(error))), "bias_k": bias,
               "abs_bias_k": abs(bias), "pearson_r": correlation, "ms_ssim": structure}
        if target_policy is not None:
            row.update(known_missing_target=False, status="scored")
        for threshold in (240, 230, 220):
            row[f"fss_{threshold}k_11px"] = fractions_skill_score(
                pred[i] <= threshold, truth[i] <= threshold, mask, 11)
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict], *, samples: list[str] | None = None,
                   leads: int = 24, target_policy: dict | None = None) -> dict:
    from src.data.final_target_policy import missing_leads
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    grouped: dict[str, dict[int, dict]] = {}
    for row in rows:
        name, hour = str(row["sample"]), int(row["lead_hour"])
        group = grouped.setdefault(name, {})
        if hour in group:
            raise ValueError(f"Duplicate date/lead row: {name}/{hour}")
        group[hour] = row
    expected = sorted(grouped) if samples is None else sorted(samples)
    if sorted(grouped) != expected or len(set(expected)) != len(expected):
        raise ValueError("Gathered evaluation samples differ from the expected unique sample list")
    valid_lead_counts = {}
    for name, group in grouped.items():
        if sorted(group) != list(range(1, leads + 1)):
            raise ValueError(f"{name}: incomplete lead rows")
        declared_missing = set(missing_leads(target_policy, name)) if target_policy is not None else set()
        scored = 0
        for hour, row in group.items():
            if hour in declared_missing:
                if (row.get("known_missing_target") is not True or row.get("status") != "known_missing_target"
                        or row.get("valid_count") != 0 or any(not np.isnan(row[m]) for m in METRICS)):
                    raise ValueError(f"{name}/{hour}: declared missing row must retain explicit unscored status")
                continue
            if row.get("known_missing_target") or row.get("status") == "known_missing_target":
                raise ValueError(f"{name}/{hour}: unexpected known-missing marker")
            if target_policy is None and row.get("status") == "no_valid_observations":
                if row.get("valid_count") != 0 or any(not np.isnan(row[m]) for m in METRICS):
                    raise ValueError(f"{name}/{hour}: unobserved frame must retain explicit unscored status")
                continue
            if row.get("valid_count", 0) <= 0:
                raise ValueError(f"{name}/{hour}: empty valid domain without approved missing target")
            if any(not np.isfinite(row[m]) for m in ("rmse_k", "mae_k", "bias_k", "abs_bias_k")):
                raise ValueError(f"{name}: nonfinite primary metric")
            scored += 1
        if scored == 0 and target_policy is not None:
            raise ValueError(f"{name}: no scored lead remains for date-equal aggregation")
        valid_lead_counts[name] = scored
    ordered = [grouped[name][hour] for name in expected for hour in range(1, leads + 1)]
    result: dict[str, Any] = {"n_dates": len(expected), "n_date_leads": len(ordered),
                              "aggregation": "mean_date(mean_lead(spatial_metric))"}
    result.update(n_scored_date_leads=sum(valid_lead_counts.values()),
                  per_date_valid_lead_counts={name: valid_lead_counts[name] for name in expected})
    if target_policy is not None:
        result["aggregation"] = "mean_date(mean_available_lead(spatial_metric))"
        result["n_known_missing_date_leads"] = len(ordered) - result["n_scored_date_leads"]
    elif result["n_scored_date_leads"] < len(ordered):
        result["aggregation"] = "mean_observed_date(mean_available_lead(spatial_metric))"
        result["n_unscored_date_leads"] = len(ordered) - result["n_scored_date_leads"]
    counts = {}
    for metric in METRICS:
        day_means = []
        finite_count = 0
        for name in expected:
            values = np.asarray([grouped[name][h][metric] for h in range(1, leads + 1)], float)
            finite = np.isfinite(values)
            finite_count += int(finite.sum())
            if finite.any():
                day_means.append(float(values[finite].mean()))
        result[metric] = float(np.mean(day_means)) if day_means else float("nan")
        counts[metric] = {"n_dates": len(day_means), "n_date_leads": finite_count}
    result["metric_support"] = counts
    result["per_lead_rows"] = ordered
    return result


def select_best_candidate(candidates: list[dict]) -> dict | None:
    eligible = [item for item in candidates
                if item.get("trained") is True and int(item.get("epoch", 0)) > 0
                and item.get("residual_enabled") is True and item.get("residual_nonzero") is True
                and np.isfinite(item.get("rmse_k", float("nan")))
                and np.isfinite(item.get("mae_k", float("nan")))]
    if not eligible:
        return None
    exact_minimum = min(float(item["rmse_k"]) for item in eligible)
    tied = [item for item in eligible if float(item["rmse_k"]) <= exact_minimum + SELECTION_TOLERANCE_K]
    return min(tied, key=lambda item: (float(item["mae_k"]), int(item["epoch"])))


@contextmanager
def preserve_rng():
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    # Each DDP worker owns one device. Accessing all CUDA RNG states would
    # initialize contexts on the other ranks' GPUs and inflate their memory.
    cuda_device = torch.cuda.current_device() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    cuda_state = torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)


def evaluation_config(config: dict) -> dict:
    output = deepcopy(config)
    output.setdefault("dataset", {}).update(
        patch_size=None, patches_per_file=1, patch_sampling="center", max_files=None,
        teacher_forcing=False, require_predicted_history_cache=False,
        predicted_history_require_manifest=False, predicted_history_max_probability=0.0)
    output.setdefault("sampling", {}).update(SAMPLING)
    return output


def write_metrics(output_dir: Path, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = metrics["per_lead_rows"]
    with (output_dir / "per_lead_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json_atomic(output_dir / "metrics.json", {k: v for k, v in metrics.items() if k != "per_lead_rows"})


def evaluate_rollout(config, model, scheduler, device, *, split="val", distributed=False,
                     output_dir=None, allow_resume_predictions=False) -> dict:
    """Shard complete dates, gather once, and return identical metrics on all ranks.

    Caller supplies EMA weights. Evaluation restores RNG/model mode so validation
    cadence cannot alter the training trajectory. An output directory enables
    immutable NPZ products; training normally requests metrics only.
    """
    import torch
    import torch.distributed as dist
    from src.data.dataset_b13cascade import B13CascadeDataset
    from src.models.diffusion import SamplingConfig
    from scripts.eval_b13cascade import predict_file
    from src.data.final_target_policy import load_policy, validate_source_record

    if distributed and not dist.is_initialized():
        raise RuntimeError("distributed evaluation requires an initialized process group")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    cfg = evaluation_config(config)
    target_policy = load_policy(cfg)
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    local_rows, names, local_error = [], [], None
    residual_max = [0.0]
    hook = None
    if hasattr(raw_model, "aerosol_residual_adapter"):
        def record_residual(_module, _inputs, output):
            value = float(output.detach().abs().max().item())
            if not np.isfinite(value):
                raise ValueError("Nonfinite residual activation")
            residual_max[0] = max(residual_max[0], value)
        hook = raw_model.aerosol_residual_adapter.register_forward_hook(record_residual)
    try:
        with preserve_rng(), torch.no_grad():
            raw_model.eval()
            dataset = B13CascadeDataset(cfg, split=split)
            names = [path.stem for path in dataset.files]
            validate_dates(names, split)
            sampling = SamplingConfig.from_config(cfg, legacy_config=False)
            for index in range(rank, len(dataset.files), world):
                path = dataset.files[index]
                if target_policy is not None:
                    validate_source_record(target_policy, path)
                target = (Path(output_dir) / "prediction_arrays" / f"{path.stem}_pred.npz"
                          if output_dir is not None else None)
                if target is not None and target.exists():
                    if not allow_resume_predictions:
                        raise FileExistsError(f"Refusing to overwrite prediction: {target}")
                    with np.load(target, allow_pickle=False) as saved:
                        local_rows.extend(prediction_rows(saved, path.stem, target_policy=target_policy)
                                          if target_policy is not None else prediction_rows(saved, path.stem))
                        recorded_residual = float(saved["residual_output_max_abs"])
                        if not np.isfinite(recorded_residual) or recorded_residual < 0:
                            raise ValueError(f"Invalid saved residual diagnostic: {target}")
                        residual_max[0] = max(residual_max[0], recorded_residual)
                    print(f"[final evaluation rank {rank}] verified resume {path.stem}", flush=True)
                    continue
                result = predict_file(dataset, index, raw_model, scheduler, device, cfg["model"],
                                      mode="autoregressive", sampling_config=sampling,
                                      cfg_cfg=cfg.get("cfg", {}))
                local_rows.extend(prediction_rows(result, path.stem, target_policy=target_policy)
                                  if target_policy is not None else prediction_rows(result, path.stem))
                if output_dir is not None:
                    arrays = Path(output_dir) / "prediction_arrays"
                    arrays.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".partial.npz")
                    result["residual_output_max_abs"] = np.asarray(residual_max[0], dtype=np.float64)
                    np.savez_compressed(temporary, **{k: v for k, v in result.items() if v is not None})
                    temporary.replace(target)
                print(f"[final evaluation rank {rank}] {split} {path.stem}", flush=True)
    except Exception as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    finally:
        if hook is not None:
            hook.remove()
        raw_model.train(was_training)
    payload = {"rows": local_rows, "samples": names, "error": local_error,
               "residual_output_max_abs": residual_max[0]}
    if distributed:
        gathered = [None] * world
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    errors = [part["error"] for part in gathered if part["error"]]
    if errors:
        raise RuntimeError("Final rollout failed: " + "; ".join(errors))
    if any(part["samples"] != gathered[0]["samples"] for part in gathered):
        raise ValueError("Different evaluation date order on distributed ranks")
    metrics = aggregate_rows([row for part in gathered for row in part["rows"]], samples=names,
                             target_policy=target_policy)
    metrics["residual_output_max_abs"] = max(part["residual_output_max_abs"] for part in gathered)
    if rank == 0 and output_dir is not None:
        write_metrics(Path(output_dir), metrics)
    return metrics

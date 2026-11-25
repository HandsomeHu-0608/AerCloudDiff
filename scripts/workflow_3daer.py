"""Prepare, train and evaluate 3DAer with an 80-epoch learning-rate schedule."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from scripts import residual_runtime as runtime
from src.utils.aerosol_residual_checkpoint import ADAPTER_ONLY_CHECKPOINT_SCHEMA, ADAPTER_ONLY_PREFLIGHT_SCHEMA
from src.utils.b13_checkpoint import checkpoint_fingerprint, write_json_atomic
from src.utils.config import load_config, save_config
from src.data.final_target_policy import policy_enabled, load_policy, build_policy_manifest, POLICY_ID

BASE_SHA = "aaca05af4ba33af4590925ac8daab79c67212cccf713d851944ebc59f1627f7e"
NORM_SHA = "e366a92b44432fdb3839ab30566db1f573ebc5b1ac08465ac927af385753105d"
AER_NORM_SHA = "fa8c40476f079fd18a8d717f3c09145bfcea8be3200eaa1cb307611c8a5c34fb"
RUN_NAME = "3DAer"
SCHEMA = "diffcloud_aer_residual_final_workflow_v2"
SEGMENTS = (15, 30, 45, 60, 80)
REFRESH = SEGMENTS[:-1]
SAMPLING = {"sampler": "ddim", "steps": 100, "eta": 0.0, "guidance_scale": 0.0,
            "clamp_mode": "per_step", "clamp_min": 0.0, "clamp_max": 1.0,
            "use_ema": True, "seed": 42,
            "seed_mode": "sample_keyed", "ensemble_members": 1}
digest = runtime.digest
read_json = runtime.read_json
run_command = runtime.run
launch4 = runtime.launch4


@contextmanager
def trajectory_lock(preparation_dir):
    """One OS-owned writer per trajectory; process exit releases the lock."""
    lock_path = Path(preparation_dir) / ".workflow.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another workflow writer holds the trajectory lock: {lock_path}") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"Another workflow writer holds the trajectory lock: {lock_path}") from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def validate_contract(config):
    """Validate the data, model, and training recipe."""
    from src.training.b13_objective import validate_3daer_loss_config
    expected = {
        "residual_workflow": {"schema": SCHEMA, "run_name": RUN_NAME,
            "baseline_checkpoint_sha256": BASE_SHA, "normalization_sha256": NORM_SHA,
            "expected_samples": 365, "expected_train_samples": 304,
            "expected_validation_samples": 30, "expected_test_samples": 31,
            "nproc_per_node": 4, "cache_refresh_epochs": list(REFRESH),
            "segment_end_epochs": list(SEGMENTS), "native_layer_indices": [0, 4, 9, 16, 24, 33],
            "max_gpu_memory_gib": 36, "preflight_steps": 2},
        "dataset": {"patch_size": [144, 184], "patches_per_file": 12,
            "cascade_horizon": 8, "cascade_start_hours": [0, 4, 8, 12, 16], "forecast_hours": 24,
            "b13_clip": [180.0, 330.0], "aod550_key": "aod550_future",
            "predicted_history_start_epoch": 1, "predicted_history_full_epoch": 1,
            "predicted_history_max_probability": 0.75, "west_edge_weighting": False,
            "loss_min_lon": None,
            "include_aerosol_full3d": True, "aerosol_full3d_cache_spatial_pool": 1,
            "aerosol_full3d_expected_species": 9, "aerosol_full3d_expected_levels": 6},
        "model": {"architecture": "grouped_aerosol_residual_st48_v2", "freeze_backbone": True,
            "residual_enabled": True, "base_channels": 160,
            "condition_embed_channels": 80, "wrf_cloud_embed_channels": 80,
            "aerosol_3d_embed_channels": 16, "aerosol_3d_species": 9,
            "aerosol_3d_levels": 6, "aerosol_3d_spatial_pool": 1,
            "timestep_embedding": "sincos", "sampler": "ddim", "ddim_eta": 0.0},
        "sampling": SAMPLING,
        "training": {"epochs": 80, "batch_size": 3, "num_workers": 0,
            "gradient_accumulation_steps": 1,
            "lr": 2e-5, "min_lr": 1e-6, "lr_warmup_epochs": 5,
            "betas": [0.9, 0.999], "weight_decay": 0.01, "grad_clip_norm": 0.5,
            "optimizer": "adamw", "amp": True, "amp_dtype": "bfloat16",
            "ema": True, "ema_validate": True, "ema_decay": 0.9995,
            "max_train_batches": None,
            "l2sp_lambda": 0.0,
            "rollout_validation_every": 5, "rollout_validation_max_files": 30,
            "rollout_validation_sampling_steps": 100,
            "rollout_guard_enabled": False, "rollout_early_stopping_enabled": False,
            "selection_rmse_tolerance_k": 1e-6, "seed": 42},
        "cfg": {"enabled": True, "dropout_prob": 0.1},
    }
    for section, values in expected.items():
        for key, wanted in values.items():
            if config.get(section, {}).get(key) != wanted:
                raise ValueError(f"Fixed 3DAer contract changed: {section}.{key}")
    validate_3daer_loss_config(config["training"])
    if config["dataset"].get("splits") != {
        "train": {"start": "2025-01-01", "end": "2025-10-31"},
        "val": {"start": "2025-11-01", "end": "2025-11-30"},
        "test": {"start": "2025-12-01", "end": "2025-12-31"},
    }:
        raise ValueError("Fixed January/October, November, December splits changed")
    if policy_enabled(config):
        required = {"target_policy_revision": "missing_targets_v1",
                    "expected_train_scored_frames": 7272, "expected_validation_scored_frames": 715,
                    "expected_test_scored_frames": 744, "expected_train_items": 18192,
                    "expected_steps_per_epoch": 1516}
        for key, value in required.items():
            if config["residual_workflow"].get(key) != value:
                raise ValueError(f"Approved missing-target contract changed: {key}")


def resolve(args):
    config = load_config(args.config)
    validate_contract(config)
    root = Path(getattr(args, "root", None) or os.environ.get("AER_RESIDUAL_FINAL_ROOT")
                or config["residual_workflow"]["root_dir"]).resolve()
    sample = Path(getattr(args, "sample_dir", None) or os.environ.get("AER_RESIDUAL_FINAL_SAMPLE_DIR")
                  or config["dataset"]["sample_dir"]).resolve()
    prep, output = root / "reports" / RUN_NAME, root / "runs" / RUN_NAME
    baseline_arg = getattr(args, "baseline", None) or os.environ.get("AER_RESIDUAL_FINAL_BASELINE")
    baseline = Path(baseline_arg) if baseline_arg else CODE / "weights/2DAer.pt"
    norm_arg = getattr(args, "normalization", None) or os.environ.get("AER_RESIDUAL_FINAL_NORMALIZATION")
    normalization = Path(norm_arg) if norm_arg else CODE / "assets/normalization/wrf_cloud.json"
    cache = root / "runs" / f"{RUN_NAME}_feature_cache"
    selection = prep / "feature_cache_selection.json"
    if selection.is_file():
        cache = Path(read_json(selection)["path"])
    requested_cache = getattr(args, "feature_cache", None) or os.environ.get("AER_RESIDUAL_FINAL_FEATURE_CACHE")
    if requested_cache:
        requested_cache = Path(requested_cache).resolve()
        if selection.is_file() and requested_cache != cache.resolve():
            raise RuntimeError("Cannot switch feature cache after preparation")
        cache = requested_cache
    paths = {"root": root, "sample": sample, "prep": prep, "output": output,
             "cache": cache.resolve(), "baseline": baseline.resolve(),
             "normalization": normalization.resolve(), "explicit_cache": bool(requested_cache),
             "verify_paper_baseline": bool(getattr(args, "verify_paper_baseline", False))}
    config["dataset"].update(sample_dir=str(sample), aerosol_full3d_cache_dir=str(paths["cache"]),
        wrf_cloud_normalization_file=str(paths["normalization"]), max_files=None,
        predicted_history_cache_dir=str(root / "runs" / f"{RUN_NAME}_history_baseline"),
        require_predicted_history_cache=True, predicted_history_require_manifest=True)
    config["output"]["output_dir"] = str(output)
    config["training"]["init_checkpoint"] = str(prep / "initial_residual_ema.pt")
    if policy_enabled(config):
        target_manifest = prep / "known_missing_targets.json"
        config["dataset"]["known_missing_targets_manifest"] = str(target_manifest)
        if target_manifest.is_file():
            config["dataset"]["known_missing_targets_manifest_sha256"] = checkpoint_fingerprint(target_manifest)
    return config, paths


def baseline_sha(paths):
    """Bind each run to the supplied checkpoint, including independently trained EMA weights."""
    return checkpoint_fingerprint(paths["baseline"])


def validate_baseline(paths):
    for key in ("baseline", "normalization"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing verified {key} asset: {paths[key]}")
    if checkpoint_fingerprint(paths["normalization"]) != NORM_SHA:
        raise RuntimeError("3DAer normalization SHA-256 mismatch")
    if paths.get("verify_paper_baseline") and baseline_sha(paths) != BASE_SHA:
        raise RuntimeError("Supplied checkpoint is not the paper's original 2DAer EMA checkpoint")
    norm = read_json(paths["normalization"])
    if norm.get("stats_files") != 304 or norm.get("stats_date_end") != "20251031":
        raise RuntimeError("WRF-cloud normalization must use the 304 training dates only")


def baseline_config(config, paths):
    result = runtime.baseline_config(config, paths)
    result["dataset"].update(max_files=None, predicted_history_cache_dir=None)
    result["model"].update(sampler="ddim", ddim_eta=0.0, sampling_steps=100)
    if policy_enabled(config):
        for key in ("known_missing_targets_policy", "known_missing_targets_manifest", "known_missing_targets_manifest_sha256"):
            result["dataset"][key] = config["dataset"][key]
    return result


def _sample_files(config, paths):
    from src.data.aerosol_vertical import sample_date
    files = sorted(paths["sample"].glob(config["dataset"]["sample_glob"]))
    dates = [sample_date(p) for p in files]
    expected = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    if sorted(dates) != expected:
        raise RuntimeError("Source coverage must be exactly the 365 daily 2025 12 UTC samples")
    return files


def validate_features(config, paths, *, full=False):
    from src.data.aerosol_vertical import CHANNEL_NAMES, SCHEMA as FEATURE_SCHEMA, SELECTED_LEVEL_INDICES
    manifest = read_json(paths["cache"] / "manifest.json")
    for key, value in {"schema": FEATURE_SCHEMA, "status": "complete", "formal_audit_qualified": True,
                       "channels": CHANNEL_NAMES, "selected_level_indices": SELECTED_LEVEL_INDICES,
                       "spatial_pool": 1, "shape_per_file": [9, 6, 24, 149, 191]}.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Unqualified feature cache: {key}")
    stats = paths["cache"] / "normalization_stats.json"
    if (not stats.is_file() or checkpoint_fingerprint(stats) != AER_NORM_SHA
            or manifest.get("normalization_stats_sha256") != AER_NORM_SHA):
        raise RuntimeError("Feature normalization fingerprint changed")
    records = manifest.get("files", {})
    files = _sample_files(config, paths)
    if {p.name for p in files} != set(records):
        raise RuntimeError("Feature cache does not cover exactly the source year")
    if full:
        print(f"[feature audit] SHA-256 source+cache scan: {len(files)} dates; cache={paths['cache']}", flush=True)
    for index, source in enumerate(files, 1):
        record = records[source.name]
        stat = source.stat()
        if stat.st_size != record.get("source_size") or stat.st_mtime_ns != record.get("source_mtime_ns"):
            raise RuntimeError(f"Feature cache source binding changed: {source.name}")
        cached = paths["cache"] / f"{source.stem}.npy"
        if not cached.is_file():
            raise FileNotFoundError(cached)
        if full and (checkpoint_fingerprint(source) != record.get("source_sha256")
                     or checkpoint_fingerprint(cached) != record.get("cache_sha256")):
            raise RuntimeError(f"Source/feature SHA-256 mismatch: {source.name}")
        if full and (index == 1 or index % 10 == 0 or index == len(files)):
            print(f"[feature audit {index}/{len(files)}] {source.name}", flush=True)
    return checkpoint_fingerprint(paths["cache"] / "manifest.json")


def migrate(config, paths):
    import torch
    from src.models.b13_factory import build_b13_model
    from src.utils.aerosol_residual_checkpoint import load_aerosol_residual_backbone
    from src.utils.b13_checkpoint import load_checkpoint_weights
    validate_baseline(paths)
    identity = {"source_checkpoint_sha256": baseline_sha(paths), "normalization_sha256": NORM_SHA,
                "model": config["model"], "schema": SCHEMA}
    target = paths["prep"] / "initial_residual_ema.pt"
    report = paths["prep"] / "migration.json"
    if report.is_file():
        saved = read_json(report)
        if saved.get("identity") != identity or not target.is_file() or checkpoint_fingerprint(target) != saved.get("checkpoint_sha256"):
            raise RuntimeError("Prepared migration changed; use a new explicit protocol version")
        return
    if target.exists():
        raise RuntimeError("Untracked migrated checkpoint; refusing to overwrite")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(42)
    source = runtime.torch_load(paths["baseline"])
    source_epoch = int(source.get("epoch", 0))
    if (source_epoch <= 0 or
            (source.get("condition_channels"), source.get("wrf_cloud_channels")) != (376, 23)):
        raise RuntimeError("Expected trained 2DAer EMA weights, condition376/WRF23")
    if paths.get("verify_paper_baseline") and source_epoch != 35:
        raise RuntimeError("The paper's 2DAer EMA checkpoint must be from epoch 35")
    reference = baseline_config(config, paths)
    for key in ("history_length", "b13_clip", "include_aod550", "include_aerosol_column_summary",
                "include_bt13_hist", "include_cloud_hist", "include_coszen", "include_daylight_mask",
                "include_wrf_meteo", "include_wrf_rad", "include_k_wrf", "include_static", "include_time",
                "cascade_horizon", "cascade_start_hours", "cascade_blend_weights"):
        if config["dataset"].get(key) != reference["dataset"].get(key):
            raise RuntimeError(f"Original 2D pathway recipe changed: {key}")
    model = build_b13_model(config["model"], 1, 376, 23).eval()
    info = load_aerosol_residual_backbone(model, source, weights="ema")
    original = build_b13_model(reference["model"], 1, 376, 23).eval()
    load_checkpoint_weights(original, source, weights="ema", allow_fallback=False)
    noisy, condition = torch.randn(1, 1, 2, 8, 8), torch.randn(1, 376, 2, 8, 8)
    cloud, aerosol = torch.randn(1, 23, 2, 8, 8), torch.randn(1, 9, 6, 2, 8, 8)
    timestep = torch.tensor([500])
    with torch.no_grad():
        expected = original(noisy, condition, timestep, cloud)
        actual = model(noisy, condition, timestep, cloud, aerosol)
    if not torch.equal(expected, actual):
        raise RuntimeError("Zero adapter does not exactly preserve original 2D EMA")
    state = model.state_dict()
    checkpoint = {"schema": "diffcloud_aer_residual_final_initial_v2", "model_state": state,
        "raw_model_state": state, "ema_model_state": state, "config": config,
        "condition_channels": 376, "wrf_cloud_channels": 23, "target_channels": 1,
        "epoch": 0, "source_epoch": source_epoch, "source_checkpoint_sha256": baseline_sha(paths),
        "normalization_sha256": NORM_SHA, "initialization_only": True,
        "baseline_fallback": False, "migration": info}
    runtime.torch_save_atomic(checkpoint, target)
    write_json_atomic(report, {"identity": identity, "source_epoch": source_epoch,
        "checkpoint_sha256": checkpoint_fingerprint(target),
        "zero_residual_max_abs_error": float((actual - expected).abs().max()), "migration": info})


def code_identity():
    names = ["scripts/workflow_3daer.py", "scripts/train_3daer.py",
        "scripts/eval_3daer.py", "scripts/report_3daer.py",
        "scripts/build_b13_predicted_history_cache.py",
        "scripts/build_aerosol_cache.py", "scripts/residual_runtime.py",
        "scripts/audit_3daer_targets.py"]
    names += [str(p.relative_to(CODE)).replace("\\", "/") for p in (CODE / "src").rglob("*.py")]
    return {name: checkpoint_fingerprint(CODE / name) for name in sorted(set(names))}


def validate_prepared(config, paths, *, full=False, require_complete=True):
    validate_contract(config)
    validate_baseline(paths)
    feature_sha = validate_features(config, paths, full=full)
    report = read_json(paths["prep"] / "migration.json")
    if report.get("identity") != {"source_checkpoint_sha256": baseline_sha(paths), "normalization_sha256": NORM_SHA,
                                  "model": config["model"], "schema": SCHEMA}:
        raise RuntimeError("Migration protocol no longer matches")
    if checkpoint_fingerprint(paths["prep"] / "initial_residual_ema.pt") != report.get("checkpoint_sha256"):
        raise RuntimeError("Migrated initialization fingerprint changed")
    identity = {"config": config, "code": code_identity(), "migration": report,
                "feature_manifest_sha256": feature_sha}
    if policy_enabled(config):
        load_policy(config)
        identity["known_missing_targets_manifest_sha256"] = checkpoint_fingerprint(config["dataset"]["known_missing_targets_manifest"])
    fingerprint = digest(identity)
    stamp = paths["prep"] / "preparation_complete.json"
    if require_complete and not stamp.is_file():
        raise RuntimeError("Run prepare first; GPU stages require its completed full-hash audit")
    if stamp.is_file() and read_json(stamp).get("fingerprint") != fingerprint:
        raise RuntimeError("Prepared code/config/data provenance changed; refuse resume")
    return fingerprint


def prepare(config, paths):
    print("[prepare] checking fixed baseline, source dates and feature cache", flush=True)
    validate_baseline(paths)
    _sample_files(config, paths)
    manifest_path = paths["cache"] / "manifest.json"
    if manifest_path.is_file() and read_json(manifest_path).get("status") == "complete":
        validate_features(config, paths, full=True)
    elif paths["explicit_cache"]:
        raise RuntimeError("Explicit feature cache must already have a complete manifest")
    else:
        reference = CODE / "assets/normalization/aerosol_3d.json"
        if checkpoint_fingerprint(reference) != AER_NORM_SHA:
            raise RuntimeError("Packaged 3DAer aerosol statistics SHA-256 mismatch")
        paths["cache"].mkdir(parents=True, exist_ok=True)
        stats = paths["cache"] / "normalization_stats.json"
        if not stats.exists():
            stats.write_bytes(reference.read_bytes())
        if checkpoint_fingerprint(stats) != AER_NORM_SHA:
            raise RuntimeError("Existing 3DAer aerosol statistics differ from the release")
        run_command([sys.executable, CODE / "scripts/build_aerosol_cache.py",
                     "--sample-dir", paths["sample"], "--output-dir", paths["cache"],
                     "--normalization-file", reference])
    write_json_atomic(paths["prep"] / "feature_cache_selection.json", {"path": str(paths["cache"]),
        "manifest_sha256": checkpoint_fingerprint(paths["cache"] / "manifest.json")})
    if policy_enabled(config):
        prepare_target_policy(config, paths)
    print("[prepare] verifying zero-residual migration", flush=True)
    migrate(config, paths)
    fingerprint = validate_prepared(config, paths, require_complete=False)
    save_config(config, paths["prep"] / "prepared_config.yaml")
    write_json_atomic(paths["prep"] / "preparation_complete.json", {"schema": SCHEMA, "fingerprint": fingerprint})
    print("[prepare] complete; next: GPU preflight and November baseline", flush=True)


def prepare_target_policy(config, paths):
    """Audit the target fields and bind the manifest to the source files."""
    from scripts.audit_3daer_targets import audit
    print("[target audit] reading 365 dates of BT13/masks/grid/time (no aerosol tensors)", flush=True)
    summary = audit(paths["sample"], paths["prep"] / "target_audit", split="all")
    manifest = build_policy_manifest(summary, read_json(paths["cache"] / "manifest.json"), paths["sample"])
    target = Path(config["dataset"]["known_missing_targets_manifest"])
    if target.is_file() and read_json(target) != manifest:
        raise RuntimeError("Bound missing-target audit changed; refusing implicit adoption")
    write_json_atomic(target, manifest)
    config["dataset"]["known_missing_targets_manifest_sha256"] = checkpoint_fingerprint(target)
    load_policy(config)
    print(f"[target audit] bound {POLICY_ID}", flush=True)


def _finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_preflight_reports(reports, steps, limit=36.0):
    if len(reports) != 4 or {int(r.get("rank", -1)) for r in reports} != set(range(4)):
        raise RuntimeError("Preflight requires one report from each of four ranks")
    for report in reports:
        if report.get("schema") != ADAPTER_ONLY_PREFLIGHT_SCHEMA:
            raise RuntimeError("Unknown preflight report schema")
        if report.get("steps") != steps:
            raise RuntimeError("Preflight did not run the required adapter steps")
        if not all(_finite_number(report.get(k)) for k in ("allocated_gib", "reserved_gib", "grad_norm")):
            raise RuntimeError("Preflight returned invalid numeric diagnostics")
        if not report.get("finite_loss") or not report.get("finite_grad") or float(report["grad_norm"]) <= 0:
            raise RuntimeError("Preflight loss/gradient failed")
        if max(float(report["allocated_gib"]), float(report["reserved_gib"])) > limit:
            raise RuntimeError(f"Preflight exceeded fixed {limit:g} GiB limit")


def preflight(config, paths, *, full=False):
    fingerprint = validate_prepared(config, paths, full=full)
    runtime.require_gpu4()
    import torch
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "gpus": [(torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i),
                  torch.cuda.get_device_properties(i).total_memory) for i in range(4)]}
    fingerprint = digest({"preparation": fingerprint, "environment": environment})
    steps = config["residual_workflow"]["preflight_steps"]
    stamp = paths["prep"] / "gpu_preflight.json"
    saved = read_json(stamp) if stamp.is_file() else {}
    if saved.get("fingerprint") == fingerprint:
        validate_preflight_reports(saved["reports"], steps)
        print("[preflight] reusing four-A100 result", flush=True)
        return
    print("[preflight] checking the adapter on four A100 GPUs", flush=True)
    smoke = copy.deepcopy(config)
    smoke["dataset"].update(require_predicted_history_cache=False,
        predicted_history_max_probability=0.0, predicted_history_cache_dir=None)
    probe_root = paths["prep"] / f"preflight_{time.time_ns()}"
    smoke["output"]["output_dir"] = str(probe_root)
    smoke_path = probe_root / "probe_config.yaml"
    save_config(smoke, smoke_path)
    launch4("scripts/train_3daer.py", ["--config", smoke_path, "--preflight",
        "--preflight-steps", str(steps), "--preflight-output", probe_root])
    reports = [read_json(probe_root / f"rank_{rank}.json") for rank in range(4)]
    validate_preflight_reports(reports, steps)
    write_json_atomic(stamp, {"fingerprint": fingerprint, "reports": reports,
                             "probe_root": str(probe_root), "formal_weights_unchanged": True})


def baseline(config, paths):
    fingerprint = validate_prepared(config, paths)
    runtime.require_gpu4()
    output = paths["prep"] / "baseline_validation"
    stamp = paths["prep"] / "baseline_complete.json"
    if stamp.is_file():
        saved = read_json(stamp)
        if saved.get("fingerprint") != fingerprint:
            raise RuntimeError("Baseline validation provenance changed")
        validate_evaluation(output, "val", 30, checkpoint_sha=baseline_sha(paths), kind="baseline_reference")
        return output
    source_config = paths["prep"] / "baseline_config.yaml"
    save_config(baseline_config(config, paths), source_config)
    print("[baseline] November 30 dates, 24h rolling inference; bound saved days resume automatically", flush=True)
    launch4("scripts/eval_3daer.py", ["--config", source_config,
        "--checkpoint", paths["baseline"], "--split", "val", "--output-dir", output])
    validate_evaluation(output, "val", 30, checkpoint_sha=baseline_sha(paths), kind="baseline_reference")
    write_json_atomic(stamp, {"fingerprint": fingerprint, "output_dir": str(output),
                             "split": "val", "n_dates": 30, "source_checkpoint_sha256": baseline_sha(paths)})
    return output


def validate_evaluation(output, split, count, *, checkpoint_sha, kind):
    output = Path(output)
    if not (output / "evaluation_complete.json").is_file():
        raise RuntimeError(f"Incomplete {split} evaluation: {output}")
    complete = read_json(output / "evaluation_complete.json")
    request = read_json(output / "evaluation_request.json")
    for key, value in {"schema": "diffcloud_aer_residual_final_evaluation_v1", "split": split,
                       "checkpoint_sha256": checkpoint_sha, "normalization_sha256": NORM_SHA,
                       "kind": kind}.items():
        if complete.get(key) != value or request.get(key) != value:
            raise RuntimeError(f"Completed evaluation provenance mismatch: {key}")
    if read_json(output / "sampling_metadata.json") != complete:
        raise RuntimeError("Evaluation completion and sampling metadata differ")
    if complete.get("weights_loaded") != "ema" or complete.get("n_dates") != count or complete.get("n_date_leads") != count * 24:
        raise RuntimeError("Evaluation did not use complete-date EMA predictions")
    for protocol in (complete.get("sampling", {}), request.get("sampling", {})):
        if any(protocol.get(key) != value for key, value in SAMPLING.items()):
            raise RuntimeError("Completed evaluation sampling protocol mismatch")
    resolved = load_config(output / "resolved_config.yaml")
    config_sha = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    if complete.get("config_sha256") != config_sha or request.get("config_sha256") != config_sha:
        raise RuntimeError("Completed evaluation resolved configuration changed")
    metrics = read_json(output / "metrics.json")
    if not all(_finite_number(metrics.get(key)) for key in ("rmse_k", "mae_k")):
        raise RuntimeError(f"Nonfinite {split} macro RMSE/MAE")
    if metrics.get("n_dates") != count or metrics.get("n_date_leads") != 24 * count:
        raise RuntimeError("Evaluation metrics have incomplete date/lead support")
    if policy_enabled(resolved):
        load_policy(resolved)
        policy_hash = checkpoint_fingerprint(resolved["dataset"]["known_missing_targets_manifest"])
        for record in (complete, request):
            if record.get("known_missing_targets_policy") != POLICY_ID or record.get("known_missing_targets_manifest_sha256") != policy_hash:
                raise RuntimeError("Evaluation missing-target policy binding differs")
        expected_scored = 715 if split == "val" else 744
        if metrics.get("n_scored_date_leads") != expected_scored or complete.get("n_scored_date_leads") != expected_scored:
            raise RuntimeError("Evaluation scored-frame support differs from the fixed policy")
    month = 11 if split == "val" else 12
    expected = {f"sample_2025{month:02d}{day:02d}12_pred.npz"
                for day in range(1, count + 1)}
    arrays = {p.name: p for p in (output / "prediction_arrays").glob("*_pred.npz")}
    hashes = complete.get("prediction_sha256", {})
    if set(arrays) != expected or set(hashes) != expected:
        raise RuntimeError(f"{split}: incomplete or unexpected evaluation dates")
    for name, path in arrays.items():
        if checkpoint_fingerprint(path) != hashes[name]:
            raise RuntimeError(f"Completed prediction array changed: {name}")
    return metrics


def history(config, paths, epoch=0):
    if epoch not in (0, *REFRESH):
        raise ValueError("History refresh is fixed at initial 2D and epochs 15/30/45/60")
    validate_prepared(config, paths)
    runtime.require_gpu4()
    if epoch == 0:
        source_checkpoint = paths["baseline"]
        recipe = baseline_config(config, paths)
        suffix = "baseline"
    else:
        source_checkpoint = paths["output"] / "checkpoints" / f"epoch_{epoch:03d}.pt"
        checkpoint = runtime.torch_load(source_checkpoint)
        if checkpoint.get("schema") != ADAPTER_ONLY_CHECKPOINT_SCHEMA or checkpoint.get("epoch") != epoch:
            raise RuntimeError("Refresh must use the exact current epoch checkpoint from this trajectory")
        if checkpoint.get("baseline_fallback") or checkpoint.get("ema_model_state") is None:
            raise RuntimeError("Refresh needs the trained EMA, never a best/fallback checkpoint")
        del checkpoint
        recipe = copy.deepcopy(config)
        suffix = f"epoch{epoch:03d}"
    source_config = paths["prep"] / f"history_source_{suffix}.yaml"
    save_config(recipe, source_config)
    output = paths["root"] / "runs" / f"{RUN_NAME}_history_{suffix}"
    runtime.build_history(config, paths, source_config, source_checkpoint, output)
    manifest = read_json(output / "manifest.json")
    expected_epoch = (int(read_json(paths["prep"] / "migration.json").get("source_epoch", 35))
                      if epoch == 0 else epoch)
    if manifest.get("source_checkpoint_epoch") != expected_epoch:
        raise RuntimeError("History manifest has the wrong source epoch")
    for key, value in SAMPLING.items():
        if manifest.get("sampling", {}).get(key) != value:
            raise RuntimeError(f"History inference protocol changed: {key}")
    if manifest.get("expected_files") != 304 or manifest.get("cache_files") != 304:
        raise RuntimeError("History refresh did not produce all 304 training dates")
    return output


def training_segments(stop_after_epoch):
    """Stop execution without shortening the learning-rate schedule."""
    if not 1 <= stop_after_epoch <= 80:
        raise ValueError("stop-after-epoch must be between 1 and 80")
    return tuple(end for end in SEGMENTS if end < stop_after_epoch) + (stop_after_epoch,)


def train(config, paths, stop_after_epoch=45):
    segments = training_segments(stop_after_epoch)
    fingerprint = validate_prepared(config, paths)
    preflight(config, paths)
    baseline(config, paths)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "final_workflow_state.json"
    if state_path.is_file():
        state = read_json(state_path)
        if state.get("fingerprint") != fingerprint:
            raise RuntimeError("Training trajectory fingerprint changed")
    else:
        if any(output.iterdir()):
            raise RuntimeError("Untracked output directory; refusing to adopt an earlier run")
        state = {"schema": SCHEMA, "fingerprint": fingerprint, "refreshes": {}}
        write_json_atomic(state_path, state)
    initial = history(config, paths, 0)
    latest = output / "checkpoints/latest.pt"
    for index, end in enumerate(segments):
        previous = segments[index - 1] if index else 0
        active = initial if previous == 0 else paths["root"] / "runs" / f"{RUN_NAME}_history_epoch{previous:03d}"
        # Complete/revalidate each preceding refresh before using its next segment.
        if previous:
            active = history(config, paths, previous)
        runtime.validate_history(config, paths, active)
        checkpoint = runtime.torch_load(latest) if latest.is_file() else None
        current = int(checkpoint.get("epoch", 0)) if checkpoint else 0
        if current > stop_after_epoch:
            raise RuntimeError("Saved training has already passed the requested stop epoch")
        if checkpoint and checkpoint.get("schema") != ADAPTER_ONLY_CHECKPOINT_SCHEMA:
            raise RuntimeError("Resume requires an adapter-only 3DAer checkpoint")
        del checkpoint
        if current < end:
            segment_config = copy.deepcopy(config)
            segment_config["dataset"]["predicted_history_cache_dir"] = str(active)
            runtime_path = paths["prep"] / f"training_through_{end:03d}.yaml"
            save_config(segment_config, runtime_path)
            arguments = ["--config", runtime_path, "--stop-after-epoch", str(end)]
            if latest.is_file():
                arguments += ["--resume", latest]
            launch4("scripts/train_3daer.py", arguments)
            finished = runtime.torch_load(latest)
            if finished.get("epoch") != end:
                raise RuntimeError(f"Trainer stopped before the required epoch {end}; no automatic second run")
            del finished
        if end in REFRESH and end < stop_after_epoch:
            refreshed = history(config, paths, end)
            state["refreshes"][str(end)] = {"path": str(refreshed), "policy": "current_epoch_ema_unconditional",
                "manifest_sha256": checkpoint_fingerprint(refreshed / "manifest.json")}
            write_json_atomic(state_path, state)
    write_json_atomic(output / "training_complete.json", {"schema": SCHEMA, "epochs": stop_after_epoch,
        "learning_rate_schedule_epochs": 80,
        "fingerprint": fingerprint, "zero_fallback_is_success": False})


def evaluate_test(config, paths, selected_epoch=30):
    complete = read_json(paths["output"] / "training_complete.json")
    completed_epoch = int(complete.get("epochs", 0))
    if complete.get("fingerprint") != validate_prepared(config, paths) or not 1 <= completed_epoch <= 80:
        raise RuntimeError("Test evaluation requires the completed requested training trajectory")
    if selected_epoch is not None and not 1 <= selected_epoch <= completed_epoch:
        raise ValueError("Evaluation epoch must have been completed by this training trajectory")
    checkpoint_name = "best_trained.pt" if selected_epoch is None else f"epoch_{selected_epoch:03d}.pt"
    target = paths["output"] / "checkpoints" / checkpoint_name
    checkpoint = runtime.torch_load(target)
    if (checkpoint.get("schema") != ADAPTER_ONLY_CHECKPOINT_SCHEMA
            or not 1 <= int(checkpoint.get("epoch", 0)) <= completed_epoch or checkpoint.get("baseline_fallback")):
        raise RuntimeError("December evaluation requires the selected trained checkpoint")
    if selected_epoch is not None and checkpoint.get("epoch") != selected_epoch:
        raise RuntimeError("The selected checkpoint filename and epoch differ")
    del checkpoint
    stamp = paths["output"] / "test_evaluation_complete.json"
    identity = {"training_fingerprint": complete["fingerprint"], "checkpoint_sha256": checkpoint_fingerprint(target)}
    if stamp.is_file():
        if read_json(stamp).get("identity") != identity:
            raise RuntimeError("December was already evaluated with another selected checkpoint")
        validate_evaluation(paths["output"] / "evaluations/december_3DAer", "test", 31,
                            checkpoint_sha=identity["checkpoint_sha256"], kind="trained_residual_candidate")
        validate_evaluation(paths["output"] / "evaluations/december_2DAer", "test", 31,
                            checkpoint_sha=baseline_sha(paths), kind="baseline_reference")
        if not (paths["output"] / "comparison/comparison_complete.json").is_file():
            raise RuntimeError("December comparison completion marker is missing")
        return
    runtime.require_gpu4()
    test_config = paths["prep"] / "test_config.yaml"
    save_config(config, test_config)
    reference_config = paths["prep"] / "baseline_config.yaml"
    save_config(baseline_config(config, paths), reference_config)
    reference = paths["output"] / "evaluations/december_2DAer"
    candidate = paths["output"] / "evaluations/december_3DAer"
    launch4("scripts/eval_3daer.py", ["--config", reference_config,
        "--checkpoint", paths["baseline"], "--split", "test", "--output-dir", reference])
    validate_evaluation(reference, "test", 31, checkpoint_sha=baseline_sha(paths), kind="baseline_reference")
    launch4("scripts/eval_3daer.py", ["--config", test_config, "--checkpoint", target,
        "--split", "test", "--output-dir", candidate])
    validate_evaluation(candidate, "test", 31, checkpoint_sha=identity["checkpoint_sha256"], kind="trained_residual_candidate")
    run_command([sys.executable, CODE / "scripts/report_3daer.py",
        "--candidate-dir", candidate, "--reference-dir", reference,
        "--reference-sha256", baseline_sha(paths),
        "--output-dir", paths["output"] / "comparison"])
    if not (paths["output"] / "comparison/comparison_complete.json").is_file():
        raise RuntimeError("Final comparison did not finish")
    write_json_atomic(stamp, {"identity": identity, "split": "test", "n_dates": 31})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate", "preflight", "baseline", "history", "train", "run"])
    parser.add_argument("--config", default=str(CODE / "configs/3DAer.yaml"))
    parser.add_argument("--root")
    parser.add_argument("--sample-dir")
    parser.add_argument("--baseline")
    parser.add_argument("--verify-paper-baseline", action="store_true",
                        help="Require the original unpublished 2DAer checkpoint fingerprint")
    parser.add_argument("--stop-after-epoch", type=int, default=45,
                        help="Completed training epochs; the learning-rate schedule remains 80 epochs")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--evaluation-epoch", type=int, default=30,
                           help="EMA epoch for December evaluation (paper: 30)")
    selection.add_argument("--use-validation-best", action="store_true",
                           help="Evaluate best_trained.pt instead of a named EMA epoch")
    parser.add_argument("--normalization")
    parser.add_argument("--feature-cache", help="Reuse an existing complete native6 cache after full hash validation")
    parser.add_argument("--history-epoch", type=int, default=0)
    parser.add_argument("--verify-full", action="store_true")
    args = parser.parse_args()
    training_segments(args.stop_after_epoch)
    config, paths = resolve(args)
    with trajectory_lock(paths["prep"]):
        if args.action == "prepare":
            prepare(config, paths)
        elif args.action == "validate":
            print(json.dumps({"fingerprint": validate_prepared(config, paths, full=args.verify_full)}, indent=2))
        elif args.action == "preflight":
            preflight(config, paths, full=args.verify_full)
        elif args.action == "baseline":
            baseline(config, paths)
        elif args.action == "history":
            history(config, paths, args.history_epoch)
        elif args.action == "train":
            train(config, paths, args.stop_after_epoch)
        else:
            if not (paths["prep"] / "preparation_complete.json").is_file():
                prepare(config, paths)
            train(config, paths, args.stop_after_epoch)
            evaluate_test(config, paths, None if args.use_validation_best else args.evaluation_epoch)


if __name__ == "__main__":
    main()

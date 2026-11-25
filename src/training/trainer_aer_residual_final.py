"""Train the 3DAer adapter with a frozen 2DAer backbone."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataset_b13cascade import B13CascadeDataset
from src.data.final_target_dataset import KnownMissingTargetDataset
from src.data.final_target_policy import POLICY_ID, load_policy
from src.models.b13_factory import build_b13_model, b13_model_kwargs
from src.models.diffusion import DDPMScheduler
from src.training.b13_objective import b13_window_loss, validate_3daer_loss_config
from src.training.ema import ExponentialMovingAverage
from src.training.trainer import cleanup_distributed, distributed_context, move_batch
from src.training.trainer_b13cascade import apply_cfg_condition_dropout, apply_cfg_aerosol_3d_dropout
from src.utils.aerosol_residual_checkpoint import ADAPTER_ONLY_CHECKPOINT_SCHEMA, ADAPTER_ONLY_PREFLIGHT_SCHEMA
from src.utils.b13_checkpoint import checkpoint_fingerprint, load_checkpoint_weights, write_json_atomic
from src.utils.config import save_config
from src.utils.seed import set_seed

SCHEMA = ADAPTER_ONLY_CHECKPOINT_SCHEMA
ARCHITECTURE = "grouped_aerosol_residual_st48_v2"
REFRESH_EPOCHS = (15, 30, 45, 60)
TOTAL_EPOCHS = 80
ADAPTER_PREFIX = "aerosol_residual_adapter."


def scheduled_lr(epoch_progress: float) -> float:
    """Continuous optimizer-step schedule given the completed epoch fraction."""
    e = min(max(float(epoch_progress), 0.0), TOTAL_EPOCHS)
    if e <= 5.0:
        return 2e-5 * e / 5.0
    return 1e-6 + (2e-5 - 1e-6) * .5 * (1.0 + math.cos(math.pi * (e - 5.0) / 75.0))


def make_optimizer(model):
    adapter = [(name, parameter) for name, parameter in model.named_parameters()
               if name.startswith(ADAPTER_PREFIX)]
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not adapter or trainable != [name for name, _ in adapter]:
        raise ValueError("3DAer requires all adapter parameters and a frozen backbone")
    group = {"params": [parameter for _, parameter in adapter], "lr": 0.0, "group_name": "adapter"}
    return torch.optim.AdamW([group], betas=(.9, .999), weight_decay=.01), [trainable]


def capture_rng_state(device=None) -> dict:
    numpy_state = np.random.get_state()
    # Tensor/primitive-only checkpoint payload remains loadable under modern
    # torch.load(weights_only=True), including the shared history-cache builder.
    numpy_serializable = (numpy_state[0], torch.from_numpy(numpy_state[1].copy()),
                          int(numpy_state[2]), int(numpy_state[3]), float(numpy_state[4]))
    state = {"python": random.getstate(), "numpy": numpy_serializable,
             "torch": torch.get_rng_state().clone()}
    if device is not None and torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def restore_rng_state(state: dict, device=None) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, keys.cpu().numpy().copy(), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if device is None or torch.device(device).type != "cuda":
            raise ValueError("A CUDA training RNG cannot resume on CPU")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


@contextmanager
def preserve_rng_state(device=None):
    state = capture_rng_state(device)
    try:
        yield
    finally:
        restore_rng_state(state, device)


def restore_adapter_ema(model, state: dict | None, device):
    ema = ExponentialMovingAverage(model, decay=.9995)
    if state is not None:
        if state["shadow"].keys() != ema.shadow.keys():
            raise ValueError("EMA parameter names differ from the 3DAer adapter")
        ema.load_state_dict(state, device=device)
    return ema


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def _atomic_copy(source: Path, destination: Path):
    temp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(destination)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def training_contract(config: dict) -> tuple[str, dict]:
    contract = {key: copy.deepcopy(config.get(key, {}))
                for key in ("dataset", "model", "training", "sampling", "cfg", "diffusion")}
    for key in ("resume", "stop_after_epoch", "progress_log_every_steps"):
        contract["training"].pop(key, None)
    contract["dataset"].pop("predicted_history_cache_dir", None)
    norm = config.get("dataset", {}).get("wrf_cloud_normalization_file")
    contract["normalization_sha256"] = checkpoint_fingerprint(norm) if norm else None
    policy = load_policy(config)
    if policy is not None:
        contract["known_missing_targets_policy"] = POLICY_ID
        contract["known_missing_targets_manifest_sha256"] = checkpoint_fingerprint(
            config["dataset"]["known_missing_targets_manifest"])
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), contract


def history_identity(config: dict) -> dict:
    directory = config.get("dataset", {}).get("predicted_history_cache_dir")
    manifest = Path(directory) / "manifest.json" if directory else None
    return {"directory": directory,
            "manifest_sha256": checkpoint_fingerprint(manifest) if manifest and manifest.is_file() else None}


def validate_resume(checkpoint: dict, config: dict, group_names, world_size: int) -> None:
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("Resume requires a checkpoint from the adapter-only 3DAer training")
    contract_hash, _ = training_contract(config)
    if checkpoint.get("training_contract_fingerprint") != contract_hash:
        raise ValueError("Training contract changed; refusing an incompatible resume")
    if checkpoint.get("optimizer_parameter_names") != group_names:
        raise ValueError("Optimizer parameter names/order changed")
    if len(checkpoint.get("rng_state_by_rank", [])) != world_size:
        raise ValueError("Resume world size differs from the saved RNG/sampler state")
    old_history = checkpoint.get("history_identity", {})
    new_history = history_identity(config)
    if old_history != new_history and int(checkpoint["epoch"]) not in REFRESH_EPOCHS:
        raise ValueError("History cache may change only at a scheduled refresh boundary")


def _validate_formal(config, context, *, preflight=False):
    if context["world_size"] != 4 or context["device"].type != "cuda":
        raise RuntimeError("Formal final training/preflight requires exactly four A100 GPUs")
    if "A100" not in torch.cuda.get_device_name(context["device"]):
        raise RuntimeError("Formal final training/preflight requires A100 GPUs")
    t, d, m = config["training"], config["dataset"], config["model"]
    if int(t.get("epochs", 80)) != 80 or int(t.get("batch_size", 3)) != 3:
        raise ValueError("Final contract requires 80 epochs and batch size 3 per GPU")
    if int(t.get("num_workers", 0)) != 0 or int(t.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("Exact epoch-boundary resume requires num_workers=0 and accumulation=1")
    if m.get("architecture") != ARCHITECTURE or int(m.get("diffusion_steps", 1000)) != 1000:
        raise ValueError("Unexpected architecture or diffusion schedule")
    if m.get("freeze_backbone") is not True:
        raise ValueError("3DAer requires a frozen backbone")
    if not m.get("residual_enabled", True) or d.get("loss_min_lon") is not None:
        raise ValueError("3DAer trains its residual on the full valid observation mask")
    validate_3daer_loss_config(t)
    if list(d.get("b13_clip", [])) != [180.0, 330.0] or not d.get("normalize_b13", True):
        raise ValueError("Final loss requires BT normalization [180,330] K")
    if not preflight:
        if (d.get("predicted_history_start_epoch"), d.get("predicted_history_full_epoch"),
            d.get("predicted_history_max_probability")) != (1, 1, .75):
            raise ValueError("Final history probability must be 0.75 from epoch one")
        if not d.get("require_predicted_history_cache"):
            raise ValueError("Formal training requires complete predicted-history cache")


def _build_runtime(config, context, *, resume=None, testing=False, dataset=None, model=None,
                   preflight=False):
    if not testing:
        _validate_formal(config, context, preflight=preflight)
    checkpoint = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    start_epoch = int(checkpoint["epoch"]) + 1 if checkpoint else 1
    set_seed(int(config.get("training", {}).get("seed", 42)))
    dataset = dataset if dataset is not None else B13CascadeDataset(config, split="train")
    policy = load_policy(config)
    if policy is not None and not isinstance(dataset, KnownMissingTargetDataset):
        dataset = KnownMissingTargetDataset(dataset, policy)
    if not testing and len(dataset.files) != 304:
        raise ValueError("Final training dataset must contain all 304 training dates")
    if not testing and policy is None and len(dataset) != 18240:
        raise ValueError("Training retains all 304 x 5 x 12 window-patch items")
    if not testing and policy is not None and (len(dataset) != 18192 or len(dataset.excluded_windows) != 4):
        raise ValueError("Missing-target training must contain 18192 items and exactly four excluded windows")
    # This shape probe consumes dataset RNG only during construction. Restore
    # the saved epoch-boundary streams after all construction/loading work.
    sample = dataset[0]
    condition_channels = int(sample["condition"].shape[0])
    wrf_channels = int(sample["wrf_cloud_condition"].shape[0])
    target_channels = int(sample["target"].shape[0])
    if model is None:
        model = build_b13_model(config["model"], target_channels, condition_channels, wrf_channels)
    model = model.to(context["device"])
    optimizer, names = make_optimizer(model)
    if checkpoint:
        validate_resume(checkpoint, config, names, context["world_size"])
        model.load_state_dict(checkpoint["raw_model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    elif config.get("training", {}).get("init_checkpoint"):
        initial = torch.load(config["training"]["init_checkpoint"], map_location="cpu", weights_only=False)
        load_checkpoint_weights(model, initial, weights="ema", strict=True, allow_fallback=False)
        del initial
    elif not testing:
        raise ValueError("Final training requires the verified expanded 2D-EMA initialization")
    ema = restore_adapter_ema(model, checkpoint["ema_state"] if checkpoint else None, context["device"])
    sampler = DistributedSampler(dataset, num_replicas=context["world_size"],
                                 rank=context["rank"], shuffle=True,
                                 seed=int(config.get("training", {}).get("seed", 42)), drop_last=False)
    loader_generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=int(config.get("training", {}).get("batch_size", 3)),
                        sampler=sampler, num_workers=0, pin_memory=context["device"].type == "cuda",
                        generator=loader_generator, drop_last=False)
    if not len(loader):
        raise ValueError("Empty training loader")
    if checkpoint:
        saved_steps = checkpoint.get("schedule_state", {}).get("steps_per_epoch")
        if saved_steps != len(loader):
            raise ValueError("Training steps per epoch changed across resume")
        if int(checkpoint.get("global_step", -1)) != int(checkpoint["epoch"]) * len(loader):
            raise ValueError("Checkpoint global step is inconsistent with its completed epochs")
    wrapped = model
    if context["distributed"]:
        kwargs = {"device_ids": [context["local_rank"]]} if context["device"].type == "cuda" else {}
        wrapped = DistributedDataParallel(model, **kwargs)
    if checkpoint:
        restore_rng_state(checkpoint["rng_state_by_rank"][context["rank"]], context["device"])
    else:
        # Model initialization is identical across ranks; training streams are
        # distinct. DataLoader/sampler randomness is explicit per epoch.
        set_seed(int(config.get("training", {}).get("seed", 42)) + context["rank"] * 1000003)
    return dict(model=model, wrapped=wrapped, optimizer=optimizer, optimizer_names=names, ema=ema,
                loader=loader, dataset=dataset, sampler=sampler, loader_generator=loader_generator,
                checkpoint=checkpoint, start_epoch=start_epoch,
                scheduler=DDPMScheduler(steps=1000, device=str(context["device"])),
                condition_channels=condition_channels, wrf_cloud_channels=wrf_channels,
                target_channels=target_channels, probe_sample=sample)


def _synchronize_failure(error, context, stage):
    """Ensure a rank-local data/loss error stops peers before DDP backward."""
    if not context["distributed"]:
        if error is not None:
            raise error
        return
    failed = torch.tensor(int(error is not None), device=context["device"], dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()):
        details = [None] * context["world_size"]
        dist.all_gather_object(details, None if error is None else
                               f"rank {context['rank']}: {type(error).__name__}: {error}")
        raise RuntimeError(f"Distributed {stage} failed: " + "; ".join(x for x in details if x))


def _synchronized_batches(loader, context):
    iterator = iter(loader)
    for step in range(1, len(loader) + 1):
        batch, error = None, None
        try:
            batch = next(iterator)
        except Exception as exc:
            error = exc
        _synchronize_failure(error, context, "training batch read")
        yield step, batch


def _train_step(runtime, batch, config, context, epoch, step, steps_in_epoch):
    device = context["device"]
    batch = move_batch(batch, device)
    optimizer = runtime["optimizer"]
    lr = scheduled_lr(epoch - 1.0 + step / steps_in_epoch)
    optimizer.param_groups[0]["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    target = batch["target"].float()
    condition, cloud, selected = apply_cfg_condition_dropout(
        batch["condition"], batch.get("wrf_cloud_condition"), batch["condition_group_ids"], config.get("cfg", {}))
    aerosol = apply_cfg_aerosol_3d_dropout(batch.get("aerosol_3d_condition"), selected, config.get("cfg", {}))
    timesteps = torch.randint(0, 1000, (target.shape[0],), device=device)
    noise = torch.randn_like(target)
    xt, noise = runtime["scheduler"].q_sample(target, timesteps, noise=noise)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = runtime["wrapped"](xt, condition, timesteps, **b13_model_kwargs(cloud, aerosol))
    error = None
    try:
        losses = b13_window_loss(prediction, noise, xt, batch, timesteps, runtime["scheduler"], epoch,
                                 config["training"], config["dataset"]["b13_clip"])
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Final loss is non-finite")
    except Exception as exc:
        error = exc
    _synchronize_failure(error, context, "training loss")
    losses["loss"].backward()
    parameters = [p for p in runtime["model"].parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
    optimizer.step()
    runtime["ema"].update(runtime["model"])
    result = {key: float(value.detach()) for key, value in losses.items()}
    result.update(grad_norm=float(grad_norm), lr_adapter=lr, n_samples=target.shape[0])
    return result


def _probe_residual(runtime, device) -> dict:
    """A real, fixed training sample proves the EMA adapter emits a residual."""
    sample = runtime["probe_sample"]
    model = runtime["model"]
    aerosol = sample["aerosol_3d_condition"].unsqueeze(0).to(device)
    cloud = sample["wrf_cloud_condition"].unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad(), preserve_rng_state(device):
        cloud_emb = model.wrf_cloud_stem(cloud)
        target_shape = (1, *sample["target"].shape)
        fixed_t = torch.full((1,), 500, dtype=torch.long, device=device)
        step_map = model._timestep_map(fixed_t, target_shape)
        residual = model.aerosol_residual_adapter(aerosol, cloud_emb, step_map)
        finite = bool(torch.isfinite(residual).all())
        maximum = float(residual.abs().max()) if finite else float("nan")
        rms = float(residual.float().square().mean().sqrt()) if finite else float("nan")
    model.train(was_training)
    return {"residual_nonzero": finite and maximum > 0.0,
            "residual_probe_absmax": maximum, "residual_probe_rms": rms}


def _gather_rng(context):
    local = capture_rng_state(context["device"])
    if not context["distributed"]:
        return [local]
    states = [None] * context["world_size"]
    dist.all_gather_object(states, local)
    return states


def _checkpoint_payload(runtime, config, epoch, global_step, candidates, best_epoch,
                        states, train_rows):
    raw = _cpu_tree(runtime["model"].state_dict())
    ema_state = runtime["ema"].state_dict()
    full_ema = dict(raw)
    full_ema.update(ema_state["shadow"])
    fingerprint, contract = training_contract(config)
    checkpoint_config = copy.deepcopy(config)
    return dict(schema=SCHEMA, checkpoint_schema_version=2, epoch=epoch, global_step=global_step,
                raw_model_state=raw, model_state=full_ema,
                ema_state=ema_state, ema_model_state=full_ema,
                optimizer_state=_cpu_tree(runtime["optimizer"].state_dict()),
                optimizer_parameter_names=runtime["optimizer_names"],
                schedule_state={"schema": "fixed_global_epoch_fraction_v1", "completed_epoch": epoch,
                                "global_step": global_step, "steps_per_epoch": len(runtime["loader"]),
                                "lr_at_boundary": scheduled_lr(epoch)},
                rng_state_by_rank=states, candidates=candidates, best_epoch=best_epoch,
                training_contract_fingerprint=fingerprint, training_contract=contract,
                history_identity=history_identity(config), train_rows=train_rows, config=checkpoint_config,
                condition_channels=runtime["condition_channels"], wrf_cloud_channels=runtime["wrf_cloud_channels"],
                target_channels=runtime["target_channels"], cfg=config.get("cfg", {}),
                diffusion=config.get("diffusion", {}), normalize_b13=True, b13_clip=[180.0, 330.0],
                normalization_sha256=contract["normalization_sha256"], trained=True,
                known_missing_targets_policy=contract.get("known_missing_targets_policy"),
                known_missing_targets_manifest_sha256=contract.get("known_missing_targets_manifest_sha256"),
                residual_enabled=True, baseline_fallback=False,
                architecture=config["model"].get("architecture"),
                residual_nonzero=next((bool(c["residual_nonzero"]) for c in reversed(candidates)
                                       if int(c["epoch"]) == epoch), None))


def _write_log(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def train_from_config(config: dict, *, resume=None, stop_after_epoch=None, testing=False,
                      dataset=None, model=None, rollout_evaluator: Callable | None = None,
                      candidate_selector: Callable | None = None) -> Path | None:
    """Production CLI uses default arguments; small injected objects are for tests."""
    config = copy.deepcopy(config)
    context = distributed_context()
    try:
        if rollout_evaluator is None or candidate_selector is None:
            from src.evaluation.aer_residual_final import evaluate_rollout, select_best_candidate
            rollout_evaluator = rollout_evaluator or evaluate_rollout
            candidate_selector = candidate_selector or select_best_candidate
        t = config.setdefault("training", {})
        resume = resume or t.get("resume")
        runtime = _build_runtime(config, context, resume=resume, testing=testing, dataset=dataset, model=model)
        checkpoint = runtime["checkpoint"] or {}
        start = runtime["start_epoch"]
        requested_end = int(stop_after_epoch or t.get("stop_after_epoch") or TOTAL_EPOCHS)
        # A new history cache is generated at each refresh boundary.
        next_boundary = next((e for e in (*REFRESH_EPOCHS, 80) if e >= start), 80)
        end = min(requested_end, TOTAL_EPOCHS, next_boundary)
        if start > end:
            raise ValueError(f"No epochs to run: checkpoint starts at {start}, stop is {end}")
        output = Path(config["output"]["output_dir"])
        checkpoint_dir = output / "checkpoints"
        if context["is_main"]:
            if not resume and (checkpoint_dir / "latest.pt").exists():
                raise FileExistsError("Existing final trajectory requires explicit --resume")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, output / "resolved_config.yaml")
        if context["distributed"]:
            dist.barrier()
        candidates = list(checkpoint.get("candidates", []))
        best_epoch = checkpoint.get("best_epoch")
        train_rows = list(checkpoint.get("train_rows", []))
        global_step = int(checkpoint.get("global_step", 0))
        runtime["checkpoint"] = None
        del checkpoint
        for epoch in range(start, end + 1):
            runtime["dataset"].set_epoch(epoch)
            runtime["sampler"].set_epoch(epoch)
            runtime["loader_generator"].manual_seed(42 + epoch * 100003 + context["rank"])
            runtime["wrapped"].train()
            sums, sample_count = {}, 0
            started = time.monotonic()
            for step, batch in _synchronized_batches(runtime["loader"], context):
                values = _train_step(runtime, batch, config, context, epoch, step, len(runtime["loader"]))
                n = values.pop("n_samples")
                sample_count += n
                global_step += 1
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.0) + value * n
                every = int(t.get("progress_log_every_steps", 100))
                if context["is_main"] and every > 0 and (step % every == 0 or step == len(runtime["loader"])):
                    print(f"epoch={epoch} step={step}/{len(runtime['loader'])} "
                          f"loss={values['loss']:.6f} grad={values['grad_norm']:.4f} "
                          f"elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)
            keys = sorted(sums)
            stats = torch.tensor([sums[k] for k in keys] + [sample_count], dtype=torch.float64, device=context["device"])
            if context["distributed"]:
                dist.all_reduce(stats)
            row = {"epoch": epoch, "global_step": global_step,
                   **{key: float(stats[i] / stats[-1]) for i, key in enumerate(keys)}}
            row["lr_adapter_end"] = scheduled_lr(epoch)
            metrics = None
            candidate_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            if epoch % 5 == 0:
                was_training = runtime["model"].training
                with preserve_rng_state(context["device"]), runtime["ema"].average_parameters(runtime["model"]):
                    metrics = rollout_evaluator(config, runtime["model"], runtime["scheduler"],
                                                context["device"], split="val",
                                                distributed=context["distributed"],
                                                output_dir=None)
                    probe = _probe_residual(runtime, context["device"])
                    if "residual_output_max_abs" in metrics:
                        value = float(metrics["residual_output_max_abs"])
                        probe["residual_nonzero"] = math.isfinite(value) and value > 0.0
                runtime["model"].train(was_training)
                if not all(math.isfinite(float(metrics[k])) for k in ("rmse_k", "mae_k")):
                    raise FloatingPointError("Full-month rollout returned non-finite primary metrics")
                candidate = dict(epoch=epoch, checkpoint=str(candidate_path), trained=True,
                                 residual_enabled=True, rmse_k=float(metrics["rmse_k"]),
                                 mae_k=float(metrics["mae_k"]), metrics=metrics, **probe)
                # All ranks use the same rank-zero probe/selection identity.
                if context["distributed"]:
                    payload = [candidate if context["is_main"] else None]
                    dist.broadcast_object_list(payload, src=0)
                    candidate = payload[0]
                candidates.append(candidate)
                selected = candidate_selector(candidates)
                best_epoch = int(selected["epoch"]) if selected is not None else None
                row.update(rollout_val_rmse_k=candidate["rmse_k"], rollout_val_mae_k=candidate["mae_k"],
                           residual_probe_rms=candidate["residual_probe_rms"], best_epoch=best_epoch)
                if context["is_main"] and metrics.get("per_lead_rows"):
                    from src.evaluation.aer_residual_final import write_metrics
                    write_metrics(output / "validation" / f"epoch_{epoch:03d}", metrics)
            train_rows.append(row)
            states = _gather_rng(context)
            if context["is_main"]:
                saved = _checkpoint_payload(runtime, config, epoch, global_step, candidates, best_epoch, states, train_rows)
                if epoch % 5 == 0:
                    _atomic_torch_save(saved, candidate_path)
                    if best_epoch is not None:
                        best_source = checkpoint_dir / f"epoch_{best_epoch:03d}.pt"
                        _atomic_copy(best_source, output / "best_trained.pt")
                        _atomic_copy(best_source, checkpoint_dir / "best_trained.pt")
                    write_json_atomic(output / "selection.json", _json_safe({
                        "schema": "diffcloud_final_selection_v1", "best_epoch": best_epoch,
                        "best_checkpoint": str(checkpoint_dir / "best_trained.pt") if best_epoch is not None else None,
                        "candidates": candidates, "baseline_fallback_allowed": False}))
                _write_log(output / "train_log.csv", train_rows)
                # latest is the transaction commit: it must never refer to a
                # candidate/selection artifact that has not reached disk yet.
                _atomic_torch_save(saved, checkpoint_dir / "latest.pt")
                del saved
            if context["distributed"]:
                dist.barrier()
        return checkpoint_dir / "best_trained.pt" if best_epoch is not None else None
    finally:
        cleanup_distributed()


def preflight_from_config(config: dict, *, steps=2, output_dir,
                          testing=False, dataset=None, model=None):
    """Isolated real-shape forward/backward probe; no formal checkpoint writes."""
    context = distributed_context()
    try:
        if int(steps) != 2:
            raise ValueError("Preflight requires two adapter optimizer steps")
        config = copy.deepcopy(config)
        if config.get("training", {}).get("resume"):
            raise ValueError("Preflight must start from the isolated initialization, not resume")
        runtime = _build_runtime(config, context, testing=testing, dataset=dataset, model=model,
                                 preflight=True)
        epoch = 1
        runtime["dataset"].set_epoch(epoch)
        runtime["sampler"].set_epoch(epoch)
        runtime["loader_generator"].manual_seed(42 + context["rank"])
        runtime["wrapped"].train()
        if context["device"].type == "cuda":
            torch.cuda.reset_peak_memory_stats(context["device"])
        reports = []
        for step, batch in _synchronized_batches(runtime["loader"], context):
            reports.append(_train_step(runtime, batch, config, context, epoch, step, len(runtime["loader"])))
            if step == int(steps):
                break
        if len(reports) != int(steps):
            raise RuntimeError("Preflight dataset did not supply two batches")
        allocated = reserved = 0.0
        if context["device"].type == "cuda":
            torch.cuda.synchronize(context["device"])
            allocated = torch.cuda.max_memory_allocated(context["device"]) / 1024**3
            reserved = torch.cuda.max_memory_reserved(context["device"]) / 1024**3
        report = dict(schema=ADAPTER_ONLY_PREFLIGHT_SCHEMA, rank=context["rank"],
                      steps=len(reports), allocated_gib=allocated, reserved_gib=reserved,
                      finite_loss=all(math.isfinite(r["loss"]) for r in reports),
                      finite_grad=all(math.isfinite(r["grad_norm"]) for r in reports),
                      grad_norm=reports[-1]["grad_norm"], step_reports=reports,
                      within_memory_budget=max(allocated, reserved) <= 36.0,
                      trainable_parameters=sum(p.numel() for p in runtime["model"].parameters() if p.requires_grad))
        destination = Path(output_dir)
        write_json_atomic(destination / f"rank_{context['rank']}.json", report)
        if not report["within_memory_budget"]:
            raise RuntimeError("Full-size preflight exceeded the fixed 36 GiB memory budget")
        if context["distributed"]:
            dist.barrier()
        return report
    finally:
        cleanup_distributed()

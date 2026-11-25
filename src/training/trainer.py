from __future__ import annotations

import csv
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, random_split
except Exception as exc:  # pragma: no cover
    torch = None
    dist = None
    DataLoader = None
    DistributedSampler = None
    DistributedDataParallel = None
    random_split = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

from src.data.dataset_diffcloud import DiffCloudDataset
from src.models.diffusion import DDPMScheduler
from src.models.unet3d import ConditionalUNet3D
from src.training.ema import ExponentialMovingAverage
from src.training.losses import cloud_probability_reconstruction_loss, log_cot_reconstruction_loss
from src.utils.config import save_config


def _require_torch() -> None:
    if torch is None:
        raise ImportError(f"PyTorch is required for training: {_TORCH_IMPORT_ERROR}")


def distributed_context() -> dict[str, Any]:
    _require_torch()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Rank-zero-only rollout validation intentionally leaves the remaining
        # ranks waiting at a collective.  Keep the historical 10-minute NCCL
        # default unless an experiment explicitly requests a longer watchdog
        # through its isolated server environment.
        timeout_seconds = int(os.environ.get("DIFFCLOUD_DDP_TIMEOUT_SECONDS", "600"))
        if timeout_seconds <= 0:
            raise ValueError("DIFFCLOUD_DDP_TIMEOUT_SECONDS must be a positive integer")
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    if os.environ.get("DIFFCLOUD_REQUIRE_CUDA", "0") == "1" and not torch.cuda.is_available():
        raise RuntimeError(
            "DIFFCLOUD_REQUIRE_CUDA=1 but PyTorch CUDA is not available. "
            "Fix the conda/PyTorch CUDA environment before launching training."
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
        "device": device,
    }


def cleanup_distributed() -> None:
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def move_batch(batch: dict[str, Any], device):
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _count_trainable_parameters(model) -> int:
    return sum(int(p.numel()) for p in _unwrap_model(model).parameters() if p.requires_grad)


def _write_split_manifest(output_dir: Path, train_ds, val_ds) -> None:
    def paths(ds) -> list[str]:
        if hasattr(ds, "files"):
            return [str(p) for p in ds.files]
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "files") and hasattr(ds, "indices"):
            return [str(ds.dataset.files[i]) for i in ds.indices]
        return []

    train_files = paths(train_ds)
    val_files = paths(val_ds)
    manifest = {
        "train_count": len(train_files),
        "val_count": len(val_files),
        "train_dataset_len": len(train_ds),
        "val_dataset_len": len(val_ds),
        "train_files": train_files,
        "val_files": val_files,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _make_datasets(config: dict[str, Any]):
    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("splits"):
        train_ds = DiffCloudDataset(config, split="train")
        val_ds = DiffCloudDataset(config, split="val")
    else:
        if dataset_cfg.get("require_explicit_splits"):
            raise ValueError("dataset.require_explicit_splits=true but dataset.splits is missing")
        dataset = DiffCloudDataset(config, split="train")
        val_size = max(1, len(dataset) // 5) if len(dataset) > 1 else 1
        train_size = max(1, len(dataset) - val_size)
        if len(dataset) > 1:
            generator = torch.Generator().manual_seed(int(config.get("training", {}).get("seed", 42)))
            train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        else:
            train_ds = dataset
            val_ds = dataset
    return train_ds, val_ds


def train_from_config(config: dict[str, Any]) -> Path:
    _require_torch()
    ctx = distributed_context()
    is_main = ctx["is_main"]
    device = ctx["device"]
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    output_dir = Path(config.get("output", {}).get("output_dir", "outputs/debug"))
    ckpt_dir = output_dir / "checkpoints"

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config_used.yaml")
        save_config(config, output_dir / "resolved_config.yaml")

    train_ds, val_ds = _make_datasets(config)
    if is_main:
        _write_split_manifest(output_dir, train_ds, val_ds)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ctx["distributed"] else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ctx["distributed"] else None
    loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, sampler=val_sampler, num_workers=0)
    first_batch = next(iter(loader))
    if is_main:
        print("First batch shapes:")
        for key in ["condition", "target", "valid_mask", "cloudy_mask", "loss_mask"]:
            print(f"  {key}: {tuple(first_batch[key].shape)}")
        print(f"  path: {first_batch['path']}")
        print(f"DDP: distributed={ctx['distributed']} world_size={ctx['world_size']} device={device}")

    condition_channels = int(first_batch["condition"].shape[1])
    target_channels = int(first_batch["target"].shape[1])
    model = ConditionalUNet3D(
        target_channels=target_channels,
        condition_channels=condition_channels,
        base_channels=int(model_cfg.get("base_channels", 16)),
        channel_mult=list(model_cfg.get("channel_mult", [1, 2, 4])),
        timestep_embedding=str(model_cfg.get("timestep_embedding", "scalar")),
    ).to(device)
    diffusion_steps = int(model_cfg.get("diffusion_steps", 100))
    model.diffusion_steps = diffusion_steps
    if ctx["distributed"]:
        ddp_kwargs = {"device_ids": [ctx["local_rank"]]} if torch.cuda.is_available() else {}
        model = DistributedDataParallel(model, **ddp_kwargs)
        model.module.diffusion_steps = diffusion_steps

    scheduler = DDPMScheduler(steps=diffusion_steps, device=str(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1.0e-4)))
    use_amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(train_cfg.get("epochs", 2))
    lambda_cloud = float(train_cfg.get("lambda_cloud", 0.0))
    lambda_cot = float(train_cfg.get("lambda_cot", 0.0))
    cloud_loss_mode = str(train_cfg.get("cloud_loss_mode", "mse"))
    cot_loss_mode = str(train_cfg.get("cot_loss_mode", "smooth_l1"))
    aux_loss_max_t_fraction = train_cfg.get("aux_loss_max_t_fraction")
    aux_loss_max_t_fraction = None if aux_loss_max_t_fraction is None else float(aux_loss_max_t_fraction)
    aux_loss_warmup_epochs = int(train_cfg.get("aux_loss_warmup_epochs", 0))
    aux_loss_start_epoch = int(train_cfg.get("aux_loss_start_epoch", 1))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    validation_seed = int(train_cfg.get("validation_seed", train_cfg.get("seed", config.get("training", {}).get("seed", 42))))
    validation_timestep_mode = str(train_cfg.get("validation_timestep_mode", "random"))
    validation_timestep_bins = int(train_cfg.get("validation_timestep_bins", 10))
    validation_noise_mode = str(train_cfg.get("validation_noise_mode", "random"))
    use_ema = bool(train_cfg.get("ema", False))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema_validate = bool(train_cfg.get("ema_validate", use_ema))
    ema = ExponentialMovingAverage(_unwrap_model(model), decay=ema_decay) if use_ema else None
    max_train_batches = train_cfg.get("max_train_batches")
    max_val_batches = train_cfg.get("max_val_batches")
    resume = train_cfg.get("resume")
    start_epoch = 1
    best_val = float("inf")

    if resume:
        ckpt = torch.load(resume, map_location=device)
        state = ckpt["model_state"]
        model_to_load = model.module if hasattr(model, "module") else model
        model_to_load.load_state_dict(ckpt.get("raw_model_state", state))
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        if ema is not None:
            if "ema_state" in ckpt:
                ema.load_state_dict(ckpt["ema_state"], device=device)
            elif "ema_model_state" in ckpt:
                ema.load_state_dict({"shadow": ckpt["ema_model_state"]}, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("val_loss", best_val))

    log_path = output_dir / "train_log.csv"
    if is_main:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=_train_log_fieldnames(),
            )
            writer.writeheader()
        print(f"Trainable parameters: {_count_trainable_parameters(model):,}")
        if ema is not None:
            print(f"EMA enabled: decay={ema_decay} validate={ema_validate}")

    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        diffusion_loss_sum = 0.0
        cloud_loss_sum = 0.0
        cot_loss_sum = 0.0
        aux_scale_sum = 0.0
        mean_t_sum = 0.0
        grad_norm_sum = 0.0
        optim_steps = 0
        train_steps = 0
        for step, batch in enumerate(loader, start=1):
            if max_train_batches is not None and step > int(max_train_batches):
                break
            batch = move_batch(batch, device)
            x0 = batch["target"]
            condition = batch["condition"]
            loss_mask = batch["loss_mask"]
            t = torch.randint(0, diffusion_steps, (x0.shape[0],), device=device, dtype=torch.long)
            xt, noise = scheduler.q_sample(x0, t)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_noise = model(xt, condition, t)
                diffusion_loss = scheduler.noise_loss(pred_noise, noise, loss_mask=loss_mask)
                cloud_aux_loss = torch.zeros((), device=device)
                cot_aux_loss = torch.zeros((), device=device)
                aux_scale = _auxiliary_loss_scale(epoch, aux_loss_start_epoch, aux_loss_warmup_epochs)
                loss = diffusion_loss
                if lambda_cloud > 0.0 or lambda_cot > 0.0:
                    pred_x0 = torch.clamp(scheduler.predict_x0(xt, t, pred_noise), 0.0, 1.0)
                    valid_mask = batch["valid_mask"]
                    cloudy_mask = batch["cloudy_mask"]
                    if aux_loss_max_t_fraction is not None:
                        t_gate = (t.float() <= float(diffusion_steps) * aux_loss_max_t_fraction).float()
                        t_gate = t_gate.view(-1, 1, 1, 1, 1)
                        valid_mask = valid_mask * t_gate
                        cloudy_mask = cloudy_mask * t_gate
                    if lambda_cloud > 0.0:
                        cloud_aux_loss = cloud_probability_reconstruction_loss(pred_x0, x0, valid_mask, mode=cloud_loss_mode)
                        loss = loss + aux_scale * lambda_cloud * cloud_aux_loss
                    if lambda_cot > 0.0:
                        cot_aux_loss = log_cot_reconstruction_loss(pred_x0, x0, cloudy_mask, mode=cot_loss_mode)
                        loss = loss + aux_scale * lambda_cot * cot_aux_loss
                backward_loss = loss / grad_accum
            scaler.scale(backward_loss).backward()
            if step % grad_accum == 0:
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(_unwrap_model(model).parameters(), grad_clip_norm)
                    grad_norm_sum += float(grad_norm.detach().cpu())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(_unwrap_model(model))
                optim_steps += 1
            train_loss_sum += float(loss.detach().cpu())
            diffusion_loss_sum += float(diffusion_loss.detach().cpu())
            cloud_loss_sum += float(cloud_aux_loss.detach().cpu())
            cot_loss_sum += float(cot_aux_loss.detach().cpu())
            aux_scale_sum += float(aux_scale)
            mean_t_sum += float(t.float().mean().detach().cpu())
            train_steps += 1

        if train_steps % grad_accum != 0:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(_unwrap_model(model).parameters(), grad_clip_norm)
                grad_norm_sum += float(grad_norm.detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(_unwrap_model(model))
            optim_steps += 1

        train_loss = _distributed_average(train_loss_sum, train_steps, device)
        diffusion_loss_avg = _distributed_average(diffusion_loss_sum, train_steps, device)
        cloud_loss_avg = _distributed_average(cloud_loss_sum, train_steps, device)
        cot_loss_avg = _distributed_average(cot_loss_sum, train_steps, device)
        aux_scale_avg = _distributed_average(aux_scale_sum, train_steps, device)
        mean_t_avg = _distributed_average(mean_t_sum, train_steps, device)
        grad_norm_avg = _distributed_average(grad_norm_sum, optim_steps, device) if optim_steps else 0.0
        if ema is not None and ema_validate:
            with ema.average_parameters(_unwrap_model(model)):
                val_loss = validate(
                    model,
                    scheduler,
                    val_loader,
                    device,
                    diffusion_steps,
                    max_batches=max_val_batches,
                    timestep_mode=validation_timestep_mode,
                    timestep_bins=validation_timestep_bins,
                    noise_mode=validation_noise_mode,
                    seed=validation_seed,
                )
        else:
            val_loss = validate(
                model,
                scheduler,
                val_loader,
                device,
                diffusion_steps,
                max_batches=max_val_batches,
                timestep_mode=validation_timestep_mode,
                timestep_bins=validation_timestep_bins,
                noise_mode=validation_noise_mode,
                seed=validation_seed,
            )
        if is_main:
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "diffusion_loss": diffusion_loss_avg,
                "cloud_aux_loss": cloud_loss_avg,
                "cot_aux_loss": cot_loss_avg,
                "aux_scale": aux_scale_avg,
                "mean_t": mean_t_avg,
                "val_loss": val_loss,
                "world_size": ctx["world_size"],
                "train_steps": train_steps,
                "optim_steps": optim_steps,
                "grad_norm": grad_norm_avg,
                "lr": optimizer.param_groups[0]["lr"],
                "ema": int(ema is not None),
            }
            with log_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=_train_log_fieldnames(),
                )
                writer.writerow(row)
            model_state = _unwrap_model(model).state_dict()
            checkpoint = {
                "model_state": model_state,
                "raw_model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
                "condition_channels": condition_channels,
                "target_channels": target_channels,
            }
            if ema is not None:
                checkpoint["ema_state"] = ema.state_dict()
                checkpoint["ema_model_state"] = {
                    name: tensor.detach().cpu() for name, tensor in ema.shadow.items()
                }
            torch.save(checkpoint, latest_path)
            torch.save(checkpoint, output_dir / "latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(checkpoint, best_path)
                torch.save(checkpoint, output_dir / "best.pt")
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    if ctx["distributed"]:
        dist.barrier()
    cleanup_distributed()
    return best_path


def _distributed_average(total: float, count: int, device) -> float:
    tensor = torch.tensor([float(total), float(count)], device=device)
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor[0] / torch.clamp(tensor[1], min=1.0)).detach().cpu())


def _train_log_fieldnames() -> list[str]:
    return [
        "epoch",
        "train_loss",
        "diffusion_loss",
        "cloud_aux_loss",
        "cot_aux_loss",
        "aux_scale",
        "mean_t",
        "val_loss",
        "world_size",
        "train_steps",
        "optim_steps",
        "grad_norm",
        "lr",
        "ema",
    ]


def _auxiliary_loss_scale(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    progress = (epoch - start_epoch + 1) / float(warmup_epochs)
    return float(max(0.0, min(1.0, progress)))


@torch.no_grad() if torch is not None else (lambda fn: fn)
def validate(
    model,
    scheduler,
    loader,
    device,
    diffusion_steps: int,
    max_batches: int | None = None,
    timestep_mode: str = "random",
    timestep_bins: int = 10,
    noise_mode: str = "random",
    seed: int = 42,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    rank = int(os.environ.get("RANK", "0"))
    for step, batch in enumerate(loader, start=1):
        if max_batches is not None and step > int(max_batches):
            break
        batch = move_batch(batch, device)
        x0 = batch["target"]
        condition = batch["condition"]
        loss_mask = batch["loss_mask"]
        generator = _make_generator(device, int(seed) + rank * 1000003 + step)
        t = _validation_timesteps(
            batch_size=x0.shape[0],
            diffusion_steps=diffusion_steps,
            step=step,
            device=device,
            mode=timestep_mode,
            bins=timestep_bins,
            generator=generator,
        )
        noise = None
        if str(noise_mode).lower() in {"fixed", "deterministic"}:
            noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=generator)
        xt, noise = scheduler.q_sample(x0, t, noise=noise)
        pred_noise = model(xt, condition, t)
        loss = scheduler.noise_loss(pred_noise, noise, loss_mask=loss_mask)
        total += float(loss.detach().cpu())
        count += 1
    return _distributed_average(total, count, device)


def _make_generator(device, seed: int):
    try:
        generator = torch.Generator(device=device)
    except Exception:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _validation_timesteps(batch_size: int, diffusion_steps: int, step: int, device, mode: str, bins: int, generator):
    mode = str(mode).lower()
    bins = max(1, int(bins))
    if mode in {"stratified", "fixed_stratified", "cycle"}:
        bin_idx = (int(step) - 1) % bins
        frac = (bin_idx + 0.5) / float(bins)
        value = int(round(frac * max(1, diffusion_steps - 1)))
        return torch.full((batch_size,), value, dtype=torch.long, device=device)
    if mode in {"mid", "fixed_mid"}:
        value = int(round(0.5 * max(1, diffusion_steps - 1)))
        return torch.full((batch_size,), value, dtype=torch.long, device=device)
    if mode in {"low", "fixed_low"}:
        value = int(round(0.25 * max(1, diffusion_steps - 1)))
        return torch.full((batch_size,), value, dtype=torch.long, device=device)
    if mode in {"high", "fixed_high"}:
        value = int(round(0.75 * max(1, diffusion_steps - 1)))
        return torch.full((batch_size,), value, dtype=torch.long, device=device)
    return torch.randint(0, diffusion_steps, (batch_size,), device=device, dtype=torch.long, generator=generator)

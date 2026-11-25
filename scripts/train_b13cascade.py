from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_b13cascade import B13CascadeDataset, FORBIDDEN_CONDITION_KEYS
from src.training.trainer_b13cascade import train_from_config
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_override_value(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        import ast

        return ast.literal_eval(value)
    except Exception:
        return value


def apply_override(config: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value, got: {override}")
    key, value = override.split("=", 1)
    parts = key.split(".")
    cur = config
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = parse_override_value(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AerCloudDiff B13 models.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--sample-dir", default=None, help="Override dataset.sample_dir.")
    parser.add_argument("--sample-glob", default=None, help="Override dataset.sample_glob.")
    parser.add_argument("--output-dir", default=None, help="Override output.output_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Limit train batches for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Limit val batches for smoke tests.")
    parser.add_argument("--resume", default=None, help="Checkpoint path for resume.")
    parser.add_argument("--stop-after-epoch", type=int, default=None, help="Pause after this epoch without changing the total LR schedule.")
    parser.add_argument("--dry-run-dataset", action="store_true", help="Only build the train dataset and print first batch shapes.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values, e.g. --set dataset.patch_size=[64,64] --set model.base_channels=16.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("dataset", {})
    config.setdefault("training", {})
    config.setdefault("output", {})
    if args.sample_dir is not None:
        config["dataset"]["sample_dir"] = args.sample_dir
    if args.sample_glob is not None:
        config["dataset"]["sample_glob"] = args.sample_glob
    if args.output_dir is not None:
        config["output"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_train_batches is not None:
        config["training"]["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        config["training"]["max_val_batches"] = args.max_val_batches
    if args.resume is not None:
        config["training"]["resume"] = args.resume
    if args.stop_after_epoch is not None:
        config["training"]["stop_after_epoch"] = args.stop_after_epoch
    for override in args.set:
        apply_override(config, override)
    set_seed(int(config.get("training", {}).get("seed", 42)))

    if args.dry_run_dataset:
        dry_run_dataset(config)
        return

    best = train_from_config(config)
    print(f"Best checkpoint: {best}")


def dry_run_dataset(config: dict) -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise ImportError(f"PyTorch is required for dataset dry-run: {exc}") from exc

    dataset = B13CascadeDataset(config, split="train")
    sample = dataset[0]
    forbidden = FORBIDDEN_CONDITION_KEYS.intersection(set(dataset.last_condition_names))
    if forbidden:
        raise RuntimeError(f"Forbidden future-label condition names found: {sorted(forbidden)}")
    loader = DataLoader(dataset, batch_size=int(config.get("training", {}).get("batch_size", 1)), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"Dataset files: {len(dataset.files)}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Condition channels: {sample['condition'].shape[0]}")
    print(f"Forbidden condition check: OK ({sorted(FORBIDDEN_CONDITION_KEYS)} absent)")
    train_cfg = config.get("training", {})
    print(f"Cold-cloud thresholds K: {train_cfg.get('cold_cloud_thresholds_k', [])}")
    print(f"Cold-cloud threshold weights: {train_cfg.get('cold_cloud_threshold_weights', [])}")
    print(f"Sampling config: {config.get('sampling', '<legacy model.* settings>')}")
    print(f"CFG config: {config.get('cfg', {'enabled': False})}")
    print(f"Diffusion extensions: {config.get('diffusion', {'offset_noise_enabled': False})}")
    print("First sample shapes:")
    for key in [
        "condition",
        "wrf_cloud_condition",
        "target",
        "target_valid_mask",
        "valid_mask",
        "loss_mask",
        "target_k",
        "boundary_prev_target",
        "boundary_valid_mask",
        "west_edge_weight_mask",
    ]:
        if key not in sample:
            continue
        print(f"  {key}: {tuple(sample[key].shape)}")
    print("First batch shapes:")
    for key in [
        "condition",
        "wrf_cloud_condition",
        "target",
        "target_valid_mask",
        "valid_mask",
        "loss_mask",
        "target_k",
        "boundary_prev_target",
        "boundary_valid_mask",
        "west_edge_weight_mask",
    ]:
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")
    print(f"  cascade_block: {batch['cascade_block']}")
    print(f"  future_hours: {batch['future_hours']}")
    print(f"  path: {batch['path']}")


if __name__ == "__main__":
    main()

"""Train noAer or 2DAer with the published staged history schedule."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.residual_runtime import launch4
from src.data.aerosol_vertical import sample_date
from src.utils.b13_checkpoint import checkpoint_fingerprint
from src.utils.config import load_config
from datetime import date, timedelta
import json


def qualifies(*args):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/qualify_history_cache.py"), *map(str, args)],
        cwd=ROOT,
    )
    if completed.returncode == 3:
        return False
    completed.check_returncode()
    return True


def build_history(config, checkpoint, samples, output):
    launch4("scripts/build_b13_predicted_history_cache.py", [
        "--source-config", config, "--source-checkpoint", checkpoint,
        "--sample-dir", samples, "--output-dir", output, "--split", "train",
        "--sampling-steps", "100", "--weights", "ema", "--seed", "42",
    ])


def train_phase(config, samples, run, end_epoch, cache):
    configured = load_config(config)
    saved_epochs = set(configured["training"].get("checkpoint_epochs", []))
    saved_epochs.update((end_epoch, {"noAer": 30, "2DAer": 35}[config.stem]))
    args = ["--config", config, "--sample-dir", samples, "--output-dir", run,
            "--stop-after-epoch", str(end_epoch), "--set",
            f"training.checkpoint_epochs={sorted(saved_epochs)}"]
    latest = run / "checkpoints/latest.pt"
    if latest.is_file():
        args += ["--resume", latest]
    elif run.exists() and any(run.iterdir()):
        raise RuntimeError(f"Existing output has no resumable checkpoint: {run}")
    if cache is None:
        args += ["--set", "dataset.require_predicted_history_cache=false",
                 "--set", "dataset.predicted_history_max_probability=0.0"]
    else:
        args += ["--set", "dataset.require_predicted_history_cache=true",
                 "--set", "dataset.predicted_history_require_manifest=true",
                 "--set", f"dataset.predicted_history_cache_dir={cache}"]
    launch4("scripts/train_b13cascade.py", args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("noAer", "2DAer"))
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--run-root", default=ROOT / "runs", type=Path)
    parser.add_argument("--history-cache", type=Path, help="Existing initial predicted-history cache for 2DAer")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Stop without changing the 80-epoch learning-rate schedule (defaults: noAer 45, 2DAer 50)")
    args = parser.parse_args()
    if args.model == "2DAer" and args.history_cache is None:
        parser.error("2DAer training requires --history-cache pointing to its initial training-history cache")
    if args.model == "noAer" and args.history_cache is not None:
        parser.error("noAer starts with observed history and does not accept --history-cache")
    samples = args.sample_dir.resolve()
    run_root = args.run_root.resolve()
    name = args.model
    stop_epoch = args.stop_after_epoch if args.stop_after_epoch is not None else {"noAer": 45, "2DAer": 50}[name]
    if not 1 <= stop_epoch <= 80:
        parser.error("--stop-after-epoch must be between 1 and 80")
    config = ROOT / f"configs/{name}.yaml"
    run = run_root / name
    if (run / f"checkpoints/epoch_{stop_epoch:03d}.pt").is_file() or (run / "early_stop.json").is_file():
        print(f"Training complete: {run}")
        return
    cfg = load_config(config)
    dates = sorted(sample_date(path) for path in samples.glob(cfg["dataset"]["sample_glob"]))
    expected = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(365)]
    if dates != expected:
        raise ValueError("Training requires the complete set of 365 daily 2025 HDF5 samples")
    provenance = json.loads((ROOT / "assets/provenance.json").read_bytes())
    norm = provenance["normalization"]["wrf_cloud"]
    if checkpoint_fingerprint(ROOT / norm["path"]) != norm["sha256"]:
        raise ValueError("WRF-cloud normalization SHA-256 mismatch")
    cache = None
    if name == "2DAer":
        cache = args.history_cache.resolve()
        if not (cache / "manifest.json").is_file():
            raise FileNotFoundError(f"Missing initial 2DAer history cache manifest: {cache}")
    for epoch in tuple(value for value in (20, 40, 80) if value < stop_epoch) + (stop_epoch,):
        checkpoint = run / f"checkpoints/epoch_{epoch:03d}.pt"
        if not checkpoint.is_file():
            train_phase(config, samples, run, epoch, cache)
        if (run / "early_stop.json").is_file():
            print(f"Training reached its configured early stop: {run}")
            return
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if epoch == stop_epoch:
            break
        if name == "noAer" and not qualifies(
            "--checkpoint", checkpoint, "--expected-epoch", epoch, "--expected-condition-channels", 347,
            "--require-ema", "--max-rmse-k", 18, "--max-abs-bias-k", 5, "--min-pearson-r", 0.60,
            "--expected-sampling-steps", 100, "--expected-max-files", 8, "--expected-seed", 42,
        ):
            cache = None if epoch == 20 else cache
            continue
        candidate = run_root / f"{name}_history_epoch{epoch:03d}"
        build_history(config, checkpoint, samples, candidate)
        if name == "2DAer" or qualifies(
            "--cache-dir", candidate, "--expected-source-checkpoint", checkpoint,
            "--cache-sampling-steps", 100, "--min-cache-files", 300,
        ):
            cache = candidate


if __name__ == "__main__":
    main()

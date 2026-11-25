"""Shared process and cache utilities for the 3DAer training workflow."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.utils.b13_checkpoint import checkpoint_fingerprint, write_json_atomic
from src.utils.config import load_config

CODE = Path(__file__).resolve().parents[1]


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(command):
    print("RUN:", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=CODE, check=True)


def baseline_config(config, paths):
    baseline = load_config(CODE / config["residual_workflow"]["baseline_config"])
    baseline["dataset"].update(sample_dir=str(paths["sample"].resolve()),
                               wrf_cloud_normalization_file=str(paths["normalization"].resolve()),
                               require_predicted_history_cache=False,
                               predicted_history_max_probability=0.0)
    baseline["sampling"] = copy.deepcopy(config["sampling"])
    return baseline


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def torch_save_atomic(checkpoint, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def require_gpu4():
    import torch
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run GPU commands inside an active Slurm allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError("This release requires exactly four allocated CUDA devices")
    for index in range(4):
        p = torch.cuda.get_device_properties(index)
        if "A100" not in p.name or p.total_memory < 38 * 1024**3:
            raise RuntimeError(f"GPU{index}: expected A100 40GB or larger, got {p.name}")


def launch4(script, arguments):
    # Static single-node rendezvous avoids elastic heartbeat failures in tmux.
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    run([sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc-per-node=4",
         "--master-addr=127.0.0.1", f"--master-port={port}", CODE / script, *arguments])


def build_history(config, paths, source_config, source_checkpoint, output):
    output = Path(output)
    source_recipe = load_config(source_config)
    source_recipe["dataset"].update(require_predicted_history_cache=False,
                                    predicted_history_cache_dir=None, predicted_history_max_probability=0.0)
    request = {"source_checkpoint_sha256":checkpoint_fingerprint(source_checkpoint),
               "feature_manifest_sha256":checkpoint_fingerprint(paths["cache"] / "manifest.json"),
               "inference_recipe":{k:source_recipe[k] for k in ("dataset","model","sampling","cfg")}}
    request["fingerprint"] = digest(request)
    request_path = output / "residual_request.json"
    pending_request = output.parent / f".{output.name}.request.json"
    if request_path.is_file():
        if read_json(request_path) != request:
            raise RuntimeError("History cache source/data/recipe changed; use a new cache directory")
    else:
        if output.exists() and any(output.iterdir()) and (not pending_request.is_file() or read_json(pending_request) != request):
            raise RuntimeError("Untracked history directory; cannot adopt legacy files into this run")
        # Keep request outside the directory until the legacy builder has
        # created its own provenance manifest (it rejects untracked files).
    if (output / "residual_source_binding.json").is_file():
        validate_history(config, paths, output)
        if not request_path.exists(): write_json_atomic(request_path, request)
        return
    if pending_request.exists() and read_json(pending_request) != request:
        raise RuntimeError("Partial history cache belongs to another request")
    write_json_atomic(pending_request, request)
    launch4("scripts/build_b13_predicted_history_cache.py", ["--source-config",source_config,
            "--source-checkpoint",source_checkpoint,"--sample-dir",paths["sample"],"--output-dir",output,
            "--split","train","--sampling-steps","100","--weights","ema","--seed","42"])
    # Explicit date parsing avoids admitting November/December through glob count alone.
    import re
    expected = {p.stem for p in paths["sample"].glob(config["dataset"]["sample_glob"])
                if "20250101" <= re.search(r"(2025\d{4})12\.h5$", p.name).group(1) <= "20251031"}
    actual = {p.stem for p in Path(output).glob("*.npz")}
    if len(expected) != 304 or actual != expected:
        raise RuntimeError("Predicted-history cache must cover exactly the 304 training dates")
    manifest = read_json(Path(output) / "manifest.json")
    if manifest.get("source_checkpoint_sha256") != checkpoint_fingerprint(source_checkpoint):
        raise RuntimeError("History source checkpoint changed")
    # Bind source data to this history cache (legacy builder binds paths only).
    write_json_atomic(Path(output) / "residual_source_binding.json", {
        "feature_manifest_sha256":checkpoint_fingerprint(paths["cache"] / "manifest.json"),
        "source_checkpoint_sha256":manifest["source_checkpoint_sha256"],
        "files":{p.name:checkpoint_fingerprint(p) for p in sorted(Path(output).glob("*.npz"))}})
    write_json_atomic(request_path, request)
    validate_history(config, paths, output)


def validate_history(config, paths, output):
    import numpy as np
    from src.data.aerosol_vertical import sample_date, is_training_date
    output = Path(output)
    binding = read_json(output / "residual_source_binding.json")
    manifest = read_json(output / "manifest.json")
    if binding.get("feature_manifest_sha256") != checkpoint_fingerprint(paths["cache"] / "manifest.json"):
        raise RuntimeError("History cache was built from different source data/features")
    if binding.get("source_checkpoint_sha256") != manifest.get("source_checkpoint_sha256"):
        raise RuntimeError("History binding/checkpoint manifest mismatch")
    if manifest.get("split") != "train" or manifest.get("weights_loaded") != "ema":
        raise RuntimeError("History must be training-only EMA predictions")
    protocol = manifest.get("sampling", {})
    for key, value in {"steps":100,"seed":42,"seed_mode":"sample_keyed","eta":0.0,
                       "guidance_scale":0.0,"ensemble_members":1,"clamp_mode":"per_step"}.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"History sampling mismatch: {key}")
    expected = {f"{p.stem}.npz" for p in paths["sample"].glob(config["dataset"]["sample_glob"])
                if is_training_date(sample_date(p))}
    if len(expected) != 304 or set(binding.get("files", {})) != expected or {p.name for p in output.glob("*.npz")} != expected:
        raise RuntimeError("History coverage differs from the 304 training samples")
    for name, fingerprint in binding["files"].items():
        path = output / name
        if checkpoint_fingerprint(path) != fingerprint:
            raise RuntimeError(f"History cache changed: {path}")
        with np.load(path, allow_pickle=False) as saved:
            if saved["pred_k"].shape != (24,149,191) or saved["valid"].shape != (24,149,191):
                raise RuntimeError(f"Bad history shape: {path}")
            if not np.isfinite(saved["pred_k"][saved["valid"].astype(bool)]).all():
                raise RuntimeError(f"Nonfinite valid history values: {path}")

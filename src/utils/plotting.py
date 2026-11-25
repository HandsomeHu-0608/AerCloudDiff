from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_line_plot(x: Sequence[float], y: Sequence[float], path: str | Path, title: str, xlabel: str, ylabel: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_hist(values: np.ndarray, path: str | Path, title: str, xlabel: str, bins: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(7, 4))
    if values.size:
        ax.hist(values, bins=bins, color="#4c78a8", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar(labels: Sequence[str], counts: Sequence[float], path: str | Path, title: str, xlabel: str, ylabel: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(x) for x in labels], counts, color="#59a14f")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sequence_grid(
    frames: np.ndarray,
    path: str | Path,
    title: str,
    max_frames: int = 8,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"Expected [T,H,W] frames, got {frames.shape}")
    n = min(max_frames, frames.shape[0])
    idx = np.linspace(0, frames.shape[0] - 1, n).round().astype(int)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j, t in enumerate(idx):
        ax = axes.ravel()[j]
        im = ax.imshow(frames[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"t={int(t)}")
        ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


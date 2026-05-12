"""Electricity dataset helper used by PatchTST_exp.ipynb.

The public notebook imports `prepare_splits` and `load_splits` from this file.
It intentionally keeps the interface small so the notebook can be re-run without
rewriting the experiment logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class ElectricitySplitInfo:
    data_path: str
    num_timesteps: int
    num_series: int
    train_end: int
    val_end: int
    lookback: int
    horizon: int
    mean: list
    std: list


class SlidingWindowDataset(Dataset):
    """Returns `(x, y)` windows with shapes `(C, L)` and `(C, H)`."""

    def __init__(self, data: np.ndarray, start: int, end: int, lookback: int, horizon: int):
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array (time, series), got shape {data.shape}")
        self.data = data.astype(np.float32, copy=False)
        self.start = int(start)
        self.end = int(end)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.length = max(0, self.end - self.start - self.lookback - self.horizon + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        i = self.start + idx
        x = self.data[i : i + self.lookback]
        y = self.data[i + self.lookback : i + self.lookback + self.horizon]
        # Model convention is (num_series, time).
        return torch.from_numpy(x.T), torch.from_numpy(y.T)


def _read_electricity_csv(data_path: str | Path) -> np.ndarray:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find Electricity CSV at {path}")

    df = pd.read_csv(path)
    # Benchmark CSVs often contain a first date column. Drop non-numeric columns.
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path}")

    arr = numeric.to_numpy(dtype=np.float32)
    # Fill occasional missing values by column means, then zeros if a column is empty.
    if np.isnan(arr).any():
        col_means = np.nanmean(arr, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
    return arr


def _make_normalized_array(data_path: str | Path, train_end: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _read_electricity_csv(data_path)
    train = raw[:train_end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (raw - mean) / std
    return normalized.astype(np.float32), mean.squeeze(0), std.squeeze(0)


def prepare_splits(
    data_path: str | Path,
    lookback: int,
    horizon: int,
    split_path: str | Path = "electricity_split.json",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """Create chronological train/val/test splits and save split metadata.

    Args:
        data_path: Path to `electricity.csv`.
        lookback: Input window length L.
        horizon: Prediction horizon H.
        split_path: JSON file used by the notebook to reload the split.
        train_ratio: Fraction of time steps used for training.
        val_ratio: Fraction of time steps used for validation.

    Returns:
        `(train_ds, val_ds, test_ds, info_dict)`.
    """
    data_path = Path(data_path).expanduser().resolve()
    raw = _read_electricity_csv(data_path)
    n_time, n_series = raw.shape
    train_end = int(n_time * train_ratio)
    val_end = int(n_time * (train_ratio + val_ratio))

    data, mean, std = _make_normalized_array(data_path, train_end)

    train_ds = SlidingWindowDataset(data, 0, train_end, lookback, horizon)
    # Include `lookback` context before each split boundary.
    val_ds = SlidingWindowDataset(data, max(0, train_end - lookback), val_end, lookback, horizon)
    test_ds = SlidingWindowDataset(data, max(0, val_end - lookback), n_time, lookback, horizon)

    info = ElectricitySplitInfo(
        data_path=str(data_path),
        num_timesteps=n_time,
        num_series=n_series,
        train_end=train_end,
        val_end=val_end,
        lookback=int(lookback),
        horizon=int(horizon),
        mean=mean.astype(float).tolist(),
        std=std.astype(float).tolist(),
    ).__dict__

    split_path = Path(split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return train_ds, val_ds, test_ds, info


def load_splits(split_path: str | Path, lookback: int | None = None, horizon: int | None = None):
    """Reload chronological splits from a JSON metadata file."""
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    info: Dict = json.loads(split_path.read_text(encoding="utf-8"))
    data_path = Path(info["data_path"])
    L = int(info["lookback"] if lookback is None else lookback)
    H = int(info["horizon"] if horizon is None else horizon)
    train_end = int(info["train_end"])
    val_end = int(info["val_end"])

    data, _, _ = _make_normalized_array(data_path, train_end)
    n_time = data.shape[0]
    train_ds = SlidingWindowDataset(data, 0, train_end, L, H)
    val_ds = SlidingWindowDataset(data, max(0, train_end - L), val_end, L, H)
    test_ds = SlidingWindowDataset(data, max(0, val_end - L), n_time, L, H)
    info = dict(info)
    info["lookback"] = L
    info["horizon"] = H
    return train_ds, val_ds, test_ds, info

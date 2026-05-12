"""
Unified PatchTST-style SSL ablation module.

This module is designed for controlled experiments on Electricity-96:
- B2: same PatchTST-style backbone trained from scratch for forecasting
- S1: random masked-patch SSL pretraining + fine-tuning
- S2: block masked-patch SSL pretraining + fine-tuning
- S3: optional overlapping-patch SSL control

Input shape throughout: (batch_size, num_series, lookback_window)
Forecast output shape: (batch_size, num_series, horizon)
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


Tensor = torch.Tensor


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = 2021, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, Tensor]:
    """Clone a model state_dict to CPU for best-checkpoint storage."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------
# Patch tokenizers
# ---------------------------------------------------------------------

class NoPadPatchTokenizer(nn.Module):
    """
    Non-overlapping/no-extra-padding tokenizer used for the paper-style SSL setup.

    Example:
        lookback=512, patch_size=12, stride=12
        num_patches = floor((512 - 12) / 12) + 1 = 42
        target_len = 12 + 12 * 41 = 504
        start = 512 - 504 = 8

    It uses the last 504 points of the 512-point input window.
    """

    def __init__(self, lookback_window: int, patch_size: int, stride: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if lookback_window < patch_size:
            raise ValueError("lookback_window must be >= patch_size")

        self.lookback_window = int(lookback_window)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.num_patches = ((self.lookback_window - self.patch_size) // self.stride) + 1
        self.target_len = self.patch_size + self.stride * (self.num_patches - 1)
        self.start = self.lookback_window - self.target_len

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,C,L), got {tuple(x.shape)}")
        x = x[:, :, self.start:]
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)


class EndPadPatchTokenizer(nn.Module):
    """
    Supervised PatchTST-style tokenizer with one extra replicate-padded tail segment.

    Example:
        lookback=512, patch_size=16, stride=8
        num_patches = floor((512 - 16) / 8) + 2 = 64

    This is useful for PatchTST/64 supervised reproduction and for the optional
    S3 overlapping-SSL control. For paper-style SSL, prefer NoPadPatchTokenizer.
    """

    def __init__(self, lookback_window: int, patch_size: int, stride: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if lookback_window < patch_size:
            raise ValueError("lookback_window must be >= patch_size")

        self.lookback_window = int(lookback_window)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.num_patches = ((self.lookback_window - self.patch_size) // self.stride) + 2

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B,C,L), got {tuple(x.shape)}")
        tail = x[:, :, -1:].expand(-1, -1, self.stride)
        x = torch.cat([x, tail], dim=-1)
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)


def build_tokenizer(
    tokenizer: str,
    lookback_window: int,
    patch_size: int,
    stride: int,
) -> nn.Module:
    tokenizer = tokenizer.lower()
    if tokenizer in {"nopad", "no_pad", "ssl"}:
        return NoPadPatchTokenizer(lookback_window, patch_size, stride)
    if tokenizer in {"endpad", "end_pad", "supervised"}:
        return EndPadPatchTokenizer(lookback_window, patch_size, stride)
    raise ValueError("tokenizer must be one of: 'nopad', 'endpad'")


# ---------------------------------------------------------------------
# PatchTST-style encoder
# ---------------------------------------------------------------------

class TSTEncoderLayer(nn.Module):
    """
    PatchTST-style Transformer encoder layer.

    Default choices are intentionally close to the official SSL scripts:
    - BatchNorm over the token dimension
    - post-norm residual blocks
    - ReLU or GELU selectable, but fixed across ablations
    - no residual-attention mechanism

    The goal is not to be a perfect clone of the repo, but to provide one
    fixed backbone for B2/S1/S2 so only the SSL objective changes.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str = "batch",
        pre_norm: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.pre_norm = bool(pre_norm)
        self.norm_type = norm_type.lower()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        if self.norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        elif self.norm_type == "layer":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            raise ValueError("norm_type must be 'batch' or 'layer'")

        act = activation.lower()
        if act == "relu":
            activation_layer = nn.ReLU()
        elif act == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def _norm(self, x: Tensor, norm: nn.Module) -> Tensor:
        if isinstance(norm, nn.BatchNorm1d):
            return norm(x.transpose(1, 2)).transpose(1, 2)
        return norm(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_norm:
            x_norm = self._norm(x, self.norm1)
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, need_weights=False)
            x = x + self.dropout_attn(attn_out)

            x_norm = self._norm(x, self.norm2)
            ff_out = self.ff(x_norm)
            x = x + self.dropout_ffn(ff_out)
            return x

        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self._norm(x + self.dropout_attn(attn_out), self.norm1)

        ff_out = self.ff(x)
        x = self._norm(x + self.dropout_ffn(ff_out), self.norm2)
        return x


class PatchTSTEncoder(nn.Module):
    """
    Channel-independent PatchTST-style encoder.

    It performs instance normalization internally, patchifies each channel,
    projects patches to tokens, adds positional embeddings, and encodes tokens
    independently per channel.
    """

    def __init__(
        self,
        lookback_window: int,
        patch_size: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str = "batch",
        pre_norm: bool = False,
        tokenizer: str = "nopad",
    ):
        super().__init__()

        self.lookback_window = int(lookback_window)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.tokenizer_name = tokenizer

        self.tokenizer = build_tokenizer(
            tokenizer=tokenizer,
            lookback_window=lookback_window,
            patch_size=patch_size,
            stride=stride,
        )
        self.num_patches = self.tokenizer.num_patches

        self.patch_proj = nn.Linear(patch_size, d_model)
        self.pos_embedding = nn.Embedding(self.num_patches, d_model)

        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                attn_dropout=attn_dropout,
                activation=activation,
                norm_type=norm_type,
                pre_norm=pre_norm,
            )
            for _ in range(n_layers)
        ])

        self.final_dropout = nn.Dropout(dropout)

    def patchify(self, x: Tensor) -> Tensor:
        return self.tokenizer(x)

    def forward(
        self,
        x: Tensor,
        patch_mask: Optional[Tensor] = None,
        mask_token: Optional[Tensor] = None,
        return_stats: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        Args:
            x: (B, C, L)
            patch_mask: optional bool tensor (B, C, N). True indicates masked patch.
            mask_token: optional tensor (d_model,). Required if patch_mask is not None.
            return_stats: if True, also return input mean/std.

        Returns:
            encoded: (B, C, N, d_model)
            target_patches: (B, C, N, patch_size), from normalized input
            optionally mean/std: both (B, C, 1)
        """
        x = x.float()
        bsz, num_series, _ = x.shape

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        x_norm = (x - mean) / std

        target_patches = self.patchify(x_norm)
        tokens = self.patch_proj(target_patches)

        if patch_mask is not None:
            if mask_token is None:
                raise ValueError("mask_token is required when patch_mask is provided")
            expected = (bsz, num_series, self.num_patches)
            if tuple(patch_mask.shape) != expected:
                raise ValueError(f"patch_mask must have shape {expected}, got {tuple(patch_mask.shape)}")
            tokens = torch.where(
                patch_mask.unsqueeze(-1),
                mask_token.view(1, 1, 1, -1).expand_as(tokens),
                tokens,
            )

        pos_ids = torch.arange(self.num_patches, device=x.device)
        pos = self.pos_embedding(pos_ids).view(1, 1, self.num_patches, self.d_model)
        tokens = tokens + pos

        tokens = tokens.reshape(bsz * num_series, self.num_patches, self.d_model)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_dropout(tokens)

        encoded = tokens.reshape(bsz, num_series, self.num_patches, self.d_model)

        if return_stats:
            return encoded, target_patches, mean, std
        return encoded, target_patches


# ---------------------------------------------------------------------
# Masks and SSL loss
# ---------------------------------------------------------------------

def random_patch_mask(
    batch_size: int,
    num_series: int,
    num_patches: int,
    mask_ratio: float,
    device: Union[str, torch.device],
) -> Tensor:
    """Exact-count independent random patch mask with shape (B, C, N)."""
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1)")

    num_mask = max(1, int(round(mask_ratio * num_patches)))
    noise = torch.rand(batch_size, num_series, num_patches, device=device)
    ids = noise.argsort(dim=-1)
    mask = torch.zeros(batch_size, num_series, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(dim=-1, index=ids[..., :num_mask], value=True)
    return mask


def random_block_patch_mask(
    batch_size: int,
    num_series: int,
    num_patches: int,
    mask_ratio: float,
    block_size: int,
    device: Union[str, torch.device],
) -> Tensor:
    """
    Approximate contiguous block mask over patch index.

    The number of masked patches can vary slightly because independently sampled
    blocks may overlap. This is acceptable for the ablation because the realized
    mask ratio is logged by the training loop.
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1)")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    num_mask_target = max(1, int(round(mask_ratio * num_patches)))
    num_blocks = max(1, int(math.ceil(num_mask_target / block_size)))

    starts = torch.randint(
        low=0,
        high=num_patches,
        size=(batch_size, num_series, num_blocks),
        device=device,
    )
    offsets = torch.arange(block_size, device=device).view(1, 1, 1, block_size)
    idx = (starts.unsqueeze(-1) + offsets) % num_patches
    idx = idx.reshape(batch_size, num_series, -1)

    mask = torch.zeros(batch_size, num_series, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(dim=-1, index=idx, value=True)
    return mask


def make_mask_fn(
    mask_type: str,
    mask_ratio: float = 0.4,
    block_size: int = 4,
) -> Callable[[int, int, int, Union[str, torch.device]], Tensor]:
    """Factory returning a mask function used by the training loops."""
    mask_type = mask_type.lower()

    if mask_type in {"random", "independent"}:
        def _fn(batch_size: int, num_series: int, num_patches: int, device):
            return random_patch_mask(batch_size, num_series, num_patches, mask_ratio, device)
        return _fn

    if mask_type in {"block", "contiguous"}:
        def _fn(batch_size: int, num_series: int, num_patches: int, device):
            return random_block_patch_mask(
                batch_size=batch_size,
                num_series=num_series,
                num_patches=num_patches,
                mask_ratio=mask_ratio,
                block_size=block_size,
                device=device,
            )
        return _fn

    raise ValueError("mask_type must be 'random' or 'block'")


def masked_patch_mse(pred_patches: Tensor, target_patches: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over masked patches only."""
    if mask.dtype != torch.bool:
        raise ValueError("mask must be bool")
    per_patch = F.mse_loss(pred_patches, target_patches, reduction="none").mean(dim=-1)
    numerator = (per_patch * mask.float()).sum()
    denominator = mask.float().sum().clamp_min(1.0)
    return numerator / denominator


@dataclass
class SSLBatchOutput:
    loss: Tensor
    pred_patches: Tensor
    target_patches: Tensor
    mask: Tensor


class PatchTSTSelfSupervised(nn.Module):
    """Masked patch reconstruction model."""

    def __init__(
        self,
        lookback_window: int,
        patch_size: int,
        stride: int,
        d_model: int = 128,
        n_heads: int = 16,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str = "batch",
        pre_norm: bool = False,
        tokenizer: str = "nopad",
        mask_ratio: float = 0.4,
    ):
        super().__init__()
        self.mask_ratio = float(mask_ratio)

        self.encoder = PatchTSTEncoder(
            lookback_window=lookback_window,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            activation=activation,
            norm_type=norm_type,
            pre_norm=pre_norm,
            tokenizer=tokenizer,
        )

        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU() if activation.lower() == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, patch_size),
        )

    def encode(self, x: Tensor) -> Tensor:
        encoded, _ = self.encoder(x)
        return encoded

    def forward(
        self,
        x: Tensor,
        patch_mask: Optional[Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> SSLBatchOutput:
        bsz, num_series, _ = x.shape
        num_patches = self.encoder.num_patches

        if patch_mask is None:
            ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
            patch_mask = random_patch_mask(
                batch_size=bsz,
                num_series=num_series,
                num_patches=num_patches,
                mask_ratio=ratio,
                device=x.device,
            )

        encoded, target_patches = self.encoder(
            x,
            patch_mask=patch_mask,
            mask_token=self.mask_token,
        )
        pred_patches = self.reconstruction_head(encoded)
        loss = masked_patch_mse(pred_patches, target_patches, patch_mask)

        return SSLBatchOutput(
            loss=loss,
            pred_patches=pred_patches,
            target_patches=target_patches,
            mask=patch_mask,
        )


class FixedPatchTSTForecastHead(nn.Module):
    """
    Forecasting head attached to a PatchTSTEncoder.

    The encoder internally instance-normalizes input windows. This head predicts
    in normalized coordinates and then maps the forecast back to the raw scale.
    """

    def __init__(self, encoder: PatchTSTEncoder, horizon: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.horizon = int(horizon)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.num_patches * encoder.d_model, horizon)

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        bsz, num_series, _ = x.shape

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)

        encoded, _ = self.encoder(x)
        encoded = self.dropout(encoded)
        encoded = encoded.reshape(
            bsz,
            num_series,
            self.encoder.num_patches * self.encoder.d_model,
        )

        y_hat_norm = self.head(encoded)
        return y_hat_norm * std + mean


def set_encoder_trainable(forecaster: FixedPatchTSTForecastHead, trainable: bool) -> None:
    for p in forecaster.encoder.parameters():
        p.requires_grad = bool(trainable)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_forecast(model: nn.Module, loader, device: Union[str, torch.device]) -> Dict[str, float]:
    model.eval()
    device = torch.device(device)
    total_mse = 0.0
    total_mae = 0.0
    total = 0

    for x, y in loader:
        x = x.float().to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        pred = model(x)

        bs = x.size(0)
        total_mse += F.mse_loss(pred, y, reduction="mean").item() * bs
        total_mae += F.l1_loss(pred, y, reduction="mean").item() * bs
        total += bs

    return {
        "mse": total_mse / max(total, 1),
        "mae": total_mae / max(total, 1),
    }


@torch.no_grad()
def last_value_baseline(loader, device: Union[str, torch.device]) -> Dict[str, float]:
    device = torch.device(device)
    total_mse = 0.0
    total_mae = 0.0
    total = 0

    for x, y in loader:
        x = x.float().to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        pred = x[:, :, -1:].repeat(1, 1, y.shape[-1])

        bs = x.size(0)
        total_mse += F.mse_loss(pred, y, reduction="mean").item() * bs
        total_mae += F.l1_loss(pred, y, reduction="mean").item() * bs
        total += bs

    return {
        "mse": total_mse / max(total, 1),
        "mae": total_mae / max(total, 1),
    }


@torch.no_grad()
def evaluate_ssl(
    model: PatchTSTSelfSupervised,
    loader,
    device: Union[str, torch.device],
    mask_fn: Optional[Callable[[int, int, int, Union[str, torch.device]], Tensor]] = None,
    fixed_seed: Optional[int] = 12345,
) -> float:
    """
    SSL validation. If fixed_seed is not None, validation masks are reproducible.
    """
    model.eval()
    device = torch.device(device)

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    if fixed_seed is not None:
        torch.manual_seed(int(fixed_seed))
        torch.cuda.manual_seed_all(int(fixed_seed))

    running = 0.0
    total = 0

    for x, _ in loader:
        x = x.float().to(device, non_blocking=True)
        bsz, num_series, _ = x.shape
        num_patches = model.encoder.num_patches

        patch_mask = None
        if mask_fn is not None:
            patch_mask = mask_fn(bsz, num_series, num_patches, x.device)

        out = model(x, patch_mask=patch_mask)
        bs = x.size(0)
        running += out.loss.item() * bs
        total += bs

    if fixed_seed is not None:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    return running / max(total, 1)


# ---------------------------------------------------------------------
# Training loops with tqdm progress and per-epoch summary prints
# ---------------------------------------------------------------------

def _amp_context(device: torch.device, amp: bool):
    amp_enabled = bool(amp and device.type == "cuda")
    return torch.amp.autocast(device_type="cuda", enabled=amp_enabled), amp_enabled


def train_forecaster_onecycle(
    model: nn.Module,
    train_loader,
    val_loader,
    device: Union[str, torch.device],
    max_epochs: int,
    max_lr: float,
    patience: Optional[int] = 10,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    pct_start: float = 0.2,
    amp: bool = True,
    desc: str = "FT",
    restore_best: bool = True,
) -> Tuple[List[Dict[str, float]], Dict[str, Tensor], float]:
    """
    Forecasting training loop:
    - Adam
    - OneCycleLR
    - AMP on CUDA
    - gradient clipping
    - early stopping by val MSE
    - tqdm progress bar
    """
    device = torch.device(device)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=max_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
    )

    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state: Dict[str, Tensor] = clone_state_dict_to_cpu(model)
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"{desc} epoch {epoch}/{max_epochs}",
            leave=True,
            mininterval=2.0,
        )

        for x, y in pbar:
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(x)
                loss = F.mse_loss(pred, y)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = x.size(0)
            running += loss.item() * bs
            total += bs
            pbar.set_postfix(train_mse=f"{running / max(total, 1):.6f}")

        train_mse = running / max(total, 1)
        val_metrics = evaluate_forecast(model, val_loader, device)
        val_mse = val_metrics["mse"]

        record = {
            "epoch": float(epoch),
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "val_mae": float(val_metrics["mae"]),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(record)

        print(
            f"{desc} epoch {epoch}/{max_epochs} | "
            f"train MSE: {train_mse:.6f} | "
            f"val MSE: {val_mse:.6f}",
            flush=True,
        )

        if val_mse < best_val:
            best_val = val_mse
            best_state = clone_state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print(f"{desc} early stopping at epoch {epoch}", flush=True)
                break

    print(f"Best {desc} val MSE: {best_val}", flush=True)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history, best_state, float(best_val)


def pretrain_ssl_onecycle(
    model: PatchTSTSelfSupervised,
    train_loader,
    val_loader,
    device: Union[str, torch.device],
    max_epochs: int,
    max_lr: float,
    mask_fn: Optional[Callable[[int, int, int, Union[str, torch.device]], Tensor]] = None,
    patience: Optional[int] = None,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    pct_start: float = 0.2,
    amp: bool = True,
    desc: str = "SSL",
    restore_best: bool = True,
    fixed_val_seed: int = 12345,
) -> Tuple[List[Dict[str, float]], Dict[str, Tensor], float]:
    """
    SSL training loop with the requested tqdm style:
        SSL epoch 1/30: 100% ... ssl_loss=...
        SSL epoch 1/30 | train SSL loss: ... | val SSL loss: ...
        Best SSL val loss: ...
    """
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
    )

    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state: Dict[str, Tensor] = clone_state_dict_to_cpu(model)
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        total = 0
        running_mask_ratio = 0.0
        mask_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"{desc} epoch {epoch}/{max_epochs}",
            leave=True,
            mininterval=2.0,
        )

        for x, _ in pbar:
            x = x.float().to(device, non_blocking=True)
            bsz, num_series, _ = x.shape
            num_patches = model.encoder.num_patches

            patch_mask = None
            if mask_fn is not None:
                patch_mask = mask_fn(bsz, num_series, num_patches, x.device)
                running_mask_ratio += patch_mask.float().mean().item()
                mask_batches += 1

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(x, patch_mask=patch_mask)
                loss = out.loss

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = x.size(0)
            running += loss.item() * bs
            total += bs
            pbar.set_postfix(ssl_loss=f"{running / max(total, 1):.6f}")

        train_ssl = running / max(total, 1)
        val_ssl = evaluate_ssl(
            model=model,
            loader=val_loader,
            device=device,
            mask_fn=mask_fn,
            fixed_seed=fixed_val_seed,
        )

        realized_mask_ratio = (
            running_mask_ratio / max(mask_batches, 1)
            if mask_fn is not None else model.mask_ratio
        )

        record = {
            "epoch": float(epoch),
            "train_ssl": float(train_ssl),
            "val_ssl": float(val_ssl),
            "lr": float(scheduler.get_last_lr()[0]),
            "mask_ratio": float(realized_mask_ratio),
        }
        history.append(record)

        print(
            f"{desc} epoch {epoch}/{max_epochs} | "
            f"train SSL loss: {train_ssl:.6f} | "
            f"val SSL loss: {val_ssl:.6f}",
            flush=True,
        )

        if val_ssl < best_val:
            best_val = val_ssl
            best_state = clone_state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print(f"{desc} early stopping at epoch {epoch}", flush=True)
                break

    print(f"Best {desc} val loss: {best_val}", flush=True)

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history, best_state, float(best_val)


# ---------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------

def save_json(path: Union[str, Path], data: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_convert)


def save_experiment_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    config: Dict,
    history: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": clone_state_dict_to_cpu(model),
        "config": copy.deepcopy(config),
        "history": history or {},
        "metrics": metrics or {},
        "extra": extra or {},
    }
    torch.save(payload, path)

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from loguru import logger
import typer
import yaml
from torch import nn
from torch.utils.data import DataLoader

from solar_forecast.nn.utils import (
    GroundPreprocessor,
    SatellitePreprocessor,
    make_dataloader,
)
from solar_forecast.config.paths import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    GROUND_DATASET_CONFIG,
    SATELLITE_DATASET_CONFIG,
    MODEL_CONFIG,
    RAW_DATA_DIR,
)

from solar_forecast.prepare import (
    load_model_cfg,
    prepare_data_and_graph,
)


from solar_forecast.nn.models.fusion import FusionModel
from solar_forecast.nn.models.satellite_only import SatelliteOnlyModel
from solar_forecast.nn.models.ground_only import GroundOnlyModel

app = typer.Typer(help="Training scripts with RANDOM DAY train/val split and Huber loss")



# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    max_grad_norm: float,
) -> float:
    model.train()
    total_loss, n = 0.0, 0

    for batch in dataloader:
        optimizer.zero_grad()

        y_hat = model(
            batch["satellite"].to(device),
            batch["ground"].to(device),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        loss = criterion(y_hat, batch["target"].to(device))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(batch["target"])
        n += len(batch["target"])

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
) -> float:
    model.eval()
    total_loss, n = 0.0, 0

    for batch in dataloader:
        y_hat = model(
            batch["satellite"].to(device),
            batch["ground"].to(device),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        loss = criterion(y_hat, batch["target"].to(device))
        total_loss += loss.item() * len(batch["target"])
        n += len(batch["target"])

    return total_loss / max(n, 1)


# ============================================================
# RANDOM DAY SPLIT
# ============================================================

def split_by_random_days(
    sat: torch.Tensor,
    ground: torch.Tensor,
    target: torch.Tensor,
    timestamps: pd.DatetimeIndex,
    val_ratio: float,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    days = pd.Index(timestamps.normalize().unique())
    n_val_days = int(len(days) * val_ratio)

    val_days = rng.choice(days, size=n_val_days, replace=False)
    val_days = pd.DatetimeIndex(val_days)

    is_val = timestamps.normalize().isin(val_days)
    is_train = ~is_val

    logger.success(
        f"Random day split → {len(days) - n_val_days} train days | {n_val_days} val days"
    )

    return (
        sat[is_train],
        ground[is_train],
        target[is_train],
        timestamps[is_train],
        sat[is_val],
        ground[is_val],
        target[is_val],
        timestamps[is_val],
    )


# ============================================================
# DATA LOADING
# ============================================================

# def load_model_cfg() -> dict:
#     with open(MODEL_CONFIG) as f:
#         return yaml.safe_load(f)


# def prepare_data_and_graph(cfg: dict):
#     from solar_forecast.train import prepare_data_and_graph as _orig
#     return _orig(cfg)


# ============================================================
# TRAIN FUSION
# ============================================================

@app.command()
def train_fusion(
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    val_ratio: float = 0.2,
    patience: int = 15,
    max_grad_norm: float = 1.0,
    lr_patience: int = 7,
    lr_factor: float = 0.5,
    huber_delta: float = 1.0,
    seed: int = 42,
    ckpt_path: Path = PROCESSED_DATA_DIR / "checkpoints" / "fusion_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_model_cfg()
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    (
        sat_tr,
        ground_tr,
        target_tr,
        ts_tr,
        sat_val,
        ground_val,
        target_val,
        ts_val,
    ) = split_by_random_days(sat, ground, target, timestamps, val_ratio, seed)

    train_loader = make_dataloader(
        sat_tr, ground_tr, target_tr, ts_tr, cfg, batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        sat_val, ground_val, target_val, ts_val, cfg, batch_size, shuffle=False
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = FusionModel(
        cfg=cfg,
        cfg_sat=cfg["model"]["satellite"],
        cfg_ground=cfg["model"]["ground"],
        cfg_fusion=cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),
    ).to(device)

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )

    best_val = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, edge_index, edge_weight, max_grad_norm
        )
        val_loss = evaluate(
            model, val_loader, criterion,
            device, edge_index, edge_weight
        )

        scheduler.step(val_loss)

        logger.info(
            f"[{epoch:03d}] train={tr_loss:.4f} val={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            logger.success(f"New best model saved → {ckpt_path}")
        else:
            wait += 1
            if wait >= patience:
                logger.warning("Early stopping triggered")
                break

    logger.success("Fusion training complete.")


# ============================================================
# TRAIN SATELLITE ONLY
# ============================================================

@app.command()
def train_satellite(
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    patience: int = 10,
    huber_delta: float = 1.0,
    seed: int = 42,
    ckpt_path: Path = PROCESSED_DATA_DIR / "checkpoints" / "satellite_only_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[SAT] Using device: {device}")

    cfg = load_model_cfg()
    sat, ground, target, timestamps, _, _ = prepare_data_and_graph(cfg)

    (
        sat_tr,
        _,
        target_tr,
        ts_tr,
        sat_val,
        _,
        target_val,
        ts_val,
    ) = split_by_random_days(sat, ground, target, timestamps, val_ratio, seed)

    train_loader = make_dataloader(
        sat_tr, torch.zeros_like(sat_tr), target_tr, ts_tr,
        cfg, batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        sat_val, torch.zeros_like(sat_val), target_val, ts_val,
        cfg, batch_size, shuffle=False
    )

    model = SatelliteOnlyModel(cfg, cfg["model"]["satellite"]).to(device)

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, None, None, 1.0
        )
        val_loss = evaluate(
            model, val_loader, criterion,
            device, None, None
        )

        logger.info(f"[{epoch:03d}] train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            logger.success("New best satellite model saved")
        else:
            wait += 1
            if wait >= patience:
                break

    logger.success("Satellite-only training complete.")


# ============================================================
# TRAIN GROUND ONLY
# ============================================================

@app.command()
def train_ground(
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    patience: int = 10,
    huber_delta: float = 1.0,
    seed: int = 42,
    ckpt_path: Path = PROCESSED_DATA_DIR / "checkpoints" / "ground_only_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[GNN] Using device: {device}")

    cfg = load_model_cfg()
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    (
        _,
        ground_tr,
        target_tr,
        ts_tr,
        _,
        ground_val,
        target_val,
        ts_val,
    ) = split_by_random_days(sat, ground, target, timestamps, val_ratio, seed)

    train_loader = make_dataloader(
        torch.zeros_like(ground_tr),
        ground_tr,
        target_tr,
        ts_tr,
        cfg,
        batch_size,
        shuffle=True,
    )
    val_loader = make_dataloader(
        torch.zeros_like(ground_val),
        ground_val,
        target_val,
        ts_val,
        cfg,
        batch_size,
        shuffle=False,
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = GroundOnlyModel(
        cfg, cfg["model"]["ground"], (edge_index, edge_weight)
    ).to(device)

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, edge_index, edge_weight, 1.0
        )
        val_loss = evaluate(
            model, val_loader, criterion,
            device, edge_index, edge_weight
        )

        logger.info(f"[{epoch:03d}] train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            logger.success("New best ground model saved")
        else:
            wait += 1
            if wait >= patience:
                break

    logger.success("Ground-only training complete.")


if __name__ == "__main__":
    app()

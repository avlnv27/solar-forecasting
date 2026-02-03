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
import matplotlib.pyplot as plt

from solar_forecast.nn.utils import make_dataloader
from solar_forecast.config.paths import (
    PROCESSED_DATA_DIR,
    GROUND_DATASET_CONFIG,
    SATELLITE_DATASET_CONFIG,
    MODEL_CONFIG,
)
from solar_forecast.nn.models.fusion import FusionModel

app = typer.Typer(help="Training with RANDOM DAY-LEVEL train/val split.")


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
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
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
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
# DATA
# ============================================================

def load_model_cfg() -> dict:
    with open(MODEL_CONFIG) as f:
        return yaml.safe_load(f)


"""
Add this function to your train.py to replace the recursive wrapper.

Replace lines 101-103 with this complete implementation.
"""

def prepare_data_and_graph(cfg: dict):
    """
    Load processed satellite, ground, target data and build adjacency graph.
    
    Returns:
        sat: Satellite data tensor (N, T, H, W)
        ground: Ground station data tensor (N, T, num_stations, features)
        target: Target CSI values tensor (N, num_stations)
        timestamps: DatetimeIndex of samples
        edge_index: Graph edge indices (2, num_edges)
        edge_weight: Graph edge weights (num_edges,)
    """
    import torch
    from torch_geometric.utils import dense_to_sparse
    
    # Get config values
    date_start = cfg["date_range"]["start"]
    date_end = cfg["date_range"]["end"]
    freq = cfg["frequency"]
    past_steps = cfg["past_timesteps"]
    future_steps = cfg["future_timesteps"]
    target_station = cfg["data"].get("target_station", "PUY")
    
    # Load satellite data
    sat_csv = PROCESSED_DATA_DIR / "satellite" / "satellite_irradiance_clean.csv"
    logger.info(f"Loading satellite data from {sat_csv}")
    sat_df = pd.read_csv(sat_csv)
    sat_df["time"] = pd.to_datetime(sat_df[cfg["data"]["sat_time_col"]])
    sat_df = sat_df.set_index("time").sort_index()
    
    # Filter date range
    sat_df = sat_df.loc[date_start:date_end]
    
    # Load ground station data
    ground_dir = PROCESSED_DATA_DIR / "ground"
    ground_files = sorted(ground_dir.glob("*_clean.csv"))
    logger.info(f"Loading {len(ground_files)} ground station files")
    
    ground_dfs = []
    station_names = []
    for gf in ground_files:
        gdf = pd.read_csv(gf)
        gdf["time"] = pd.to_datetime(gdf["time"])
        gdf = gdf.set_index("time").sort_index()
        gdf = gdf.loc[date_start:date_end]
        ground_dfs.append(gdf)
        
        # Extract station code from filename
        station_code = gf.stem.split("_")[0].upper()
        station_names.append(station_code)
    
    logger.info(f"Stations: {station_names}")
    
    # Find target station index
    if target_station:
        try:
            target_idx = station_names.index(target_station.upper())
        except ValueError:
            logger.warning(f"Target station {target_station} not found, using first station")
            target_idx = 0
    else:
        target_idx = 0
    
    logger.info(f"Target station: {station_names[target_idx]} (index {target_idx})")
    
    # Merge all stations on common timestamps
    common_times = ground_dfs[0].index
    for gdf in ground_dfs[1:]:
        common_times = common_times.intersection(gdf.index)
    
    # Also intersect with satellite times
    common_times = common_times.intersection(sat_df.index)
    
    logger.info(f"Common timestamps: {len(common_times)}")
    
    # Resample to desired frequency if needed
    if freq != "10min":  # Assuming data is already at 10min
        common_times = pd.date_range(
            start=common_times[0],
            end=common_times[-1],
            freq=freq
        )
    
    # Extract CSI values
    ground_csi_col = cfg["data"]["ground_value_col"]
    sat_csi_col = cfg["data"]["sat_value_col"]
    
    # Build arrays
    num_stations = len(ground_dfs)
    num_times = len(common_times)
    
    # Ground: (num_times, num_stations)
    ground_array = np.zeros((num_times, num_stations), dtype=np.float32)
    for i, gdf in enumerate(ground_dfs):
        ground_array[:, i] = gdf.loc[common_times, ground_csi_col].values
    
    # Satellite: (num_times, H, W) - assuming satellite data has spatial dimensions
    # For now, treat as scalar and expand later
    sat_array = sat_df.loc[common_times, sat_csi_col].values.astype(np.float32)
    
    # Target: same as target station ground data
    target_array = ground_array[:, target_idx].copy()
    
    # Create sliding windows
    valid_starts = []
    for i in range(len(common_times) - past_steps - future_steps + 1):
        valid_starts.append(i)
    
    num_samples = len(valid_starts)
    logger.info(f"Creating {num_samples} samples with {past_steps} past + {future_steps} future steps")
    
    # Satellite windows: (N, T, H, W)
    # For now, assuming 11x11 spatial grid (you may need to adjust)
    H, W = 11, 11
    sat_windows = np.zeros((num_samples, past_steps, H, W), dtype=np.float32)
    for n, start_idx in enumerate(valid_starts):
        for t in range(past_steps):
            # Replicate scalar to spatial grid (placeholder)
            sat_windows[n, t, :, :] = sat_array[start_idx + t]
    
    # Ground windows: (N, T, num_stations, 1)
    ground_windows = np.zeros((num_samples, past_steps, num_stations, 1), dtype=np.float32)
    for n, start_idx in enumerate(valid_starts):
        ground_windows[n, :, :, 0] = ground_array[start_idx:start_idx + past_steps, :]
    
    # Target windows: (N, future_steps, num_stations)
    target_windows = np.zeros((num_samples, future_steps, num_stations), dtype=np.float32)
    for n, start_idx in enumerate(valid_starts):
        target_windows[n, :, :] = ground_array[
            start_idx + past_steps:start_idx + past_steps + future_steps, :
        ]
    
    # For now, we predict all stations (not just target)
    # Reshape target to (N, num_stations * future_steps) or keep as (N, future_steps, num_stations)
    # Let's keep it simple: predict only target station
    target_windows = target_windows[:, :, target_idx]  # (N, future_steps)
    
    # Sample timestamps (start of prediction window)
    sample_timestamps = pd.DatetimeIndex([
        common_times[start_idx + past_steps] for start_idx in valid_starts
    ])
    
    # Build adjacency graph (simple distance-based)
    # Get station locations (placeholder - you may have real lat/lon)
    # For now, use fully connected graph
    adjacency_matrix = np.ones((num_stations, num_stations), dtype=np.float32)
    np.fill_diagonal(adjacency_matrix, 0)  # No self-loops
    
    adjacency_tensor = torch.from_numpy(adjacency_matrix)
    edge_index, edge_weight = dense_to_sparse(adjacency_tensor)
    
    # Convert to tensors
    sat_tensor = torch.from_numpy(sat_windows)
    ground_tensor = torch.from_numpy(ground_windows)
    target_tensor = torch.from_numpy(target_windows)
    
    logger.success(f"Data loaded: sat={sat_tensor.shape}, ground={ground_tensor.shape}, target={target_tensor.shape}")
    logger.info(f"Graph: {edge_index.shape[1]} edges between {num_stations} nodes")
    
    return sat_tensor, ground_tensor, target_tensor, sample_timestamps, edge_index, edge_weight



def split_by_random_days(
    sat, ground, target, timestamps, val_ratio: float, seed: int = 42
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
# TRAIN FUSION - FIXED: SAVES FULL CONFIG
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
    huber_delta: float = 0.5,
    seed: int = 42,
    ckpt_path: Path = PROCESSED_DATA_DIR / "checkpoints" / "fusion_best.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_model_cfg()
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    (
        sat_tr, ground_tr, target_tr, ts_tr,
        sat_val, ground_val, target_val, ts_val,
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

    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )

    # LOSS HISTORY
    train_losses, val_losses = [], []

    best_val = float("inf")
    best_epoch = 0
    wait = 0

    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, edge_index, edge_weight, max_grad_norm
        )
        val_loss = evaluate(
            model, val_loader, criterion, device, edge_index, edge_weight
        )

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"[{epoch:03d}] train={tr_loss:.4f} val={val_loss:.4f} "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            
            # FIXED: Save complete checkpoint with config
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,  # ← CRITICAL: Save config!
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "hyperparams": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "huber_delta": huber_delta,
                    "max_grad_norm": max_grad_norm,
                }
            }, ckpt_path)
            
            logger.success(f"✓ New best model saved → {ckpt_path} (epoch {epoch})")
        else:
            wait += 1
            if wait >= patience:
                logger.warning(f"Early stopping triggered at epoch {epoch}")
                break

    logger.info("=" * 70)
    logger.success(f"Training completed!")
    logger.success(f"Best epoch: {best_epoch} | Best val loss: {best_val:.4f}")
    logger.info("=" * 70)

    # ============================================================
    # SAVE LOSS CSV + PLOT
    # ============================================================

    out_dir = PROCESSED_DATA_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_loss = pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
    })

    csv_path = out_dir / "training_losses_fusion.csv"
    df_loss.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df_loss["epoch"], df_loss["train_loss"], label="Train", linewidth=2)
    plt.plot(df_loss["epoch"], df_loss["val_loss"], label="Validation", linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss – Fusion Model", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = out_dir / "training_loss_curve_fusion.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.success(f"Loss CSV saved → {csv_path}")
    logger.success(f"Loss plot saved → {fig_path}")
    logger.success(f"Checkpoint saved → {ckpt_path}")


if __name__ == "__main__":
    app()
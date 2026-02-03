"""
Retrain Satellite Only model with best hyperparameters from Optuna.

Run:
python -m solar_forecast.retrain_satellite_best --epochs 200
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger
import typer
import yaml
from torch import nn
import matplotlib.pyplot as plt

from solar_forecast.nn.utils import make_dataloader
from solar_forecast.config.paths import MODELS_DIR
from solar_forecast.nn.models.satellite_only import SatelliteOnlyModel
from solar_forecast.train_huber_randomday import (
    load_model_cfg,
    prepare_data_and_graph,
    split_by_random_days,
)

app = typer.Typer(help="Retrain Satellite Only with best Optuna hyperparameters")


def build_config_from_params(base_cfg: dict, params: dict) -> dict:
    import copy
    cfg = copy.deepcopy(base_cfg)

    if "sat_feature_dim" in params:
        cfg["model"]["satellite"]["feature_dim"] = params["sat_feature_dim"]
    if "sat_temporal_layers" in params:
        cfg["model"]["satellite"]["temporal_layers"] = params["sat_temporal_layers"]
    if "sat_conv1_channels" in params:
        cfg["model"]["satellite"]["conv1_channels"] = params["sat_conv1_channels"]
    if "sat_conv2_channels" in params:
        cfg["model"]["satellite"]["conv2_channels"] = params["sat_conv2_channels"]
    if "sat_head_hidden_dim" in params:
        cfg["model"]["satellite"]["head_hidden_dim"] = params["sat_head_hidden_dim"]
    if "sat_use_attention" in params:
        cfg["model"]["satellite"]["use_attention"] = params["sat_use_attention"]
    if "sat_attn_dim" in params:
        cfg["model"]["satellite"]["attn_dim"] = params["sat_attn_dim"]

    return cfg


def train_one_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss, n = 0.0, 0

    for batch in dataloader:
        optimizer.zero_grad()
        y_hat = model(batch["satellite"].to(device))
        loss = criterion(y_hat, batch["target"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * len(batch["target"])
        n += len(batch["target"])

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0

    for batch in dataloader:
        y_hat = model(batch["satellite"].to(device))
        loss = criterion(y_hat, batch["target"].to(device))
        total_loss += loss.item() * len(batch["target"])
        n += len(batch["target"])

    return total_loss / max(n, 1)


@app.command()
def retrain(
    epochs: int = 200,
    val_ratio: float = 0.2,
    patience: int = 25,
    seed: int = 42,
):
    """Retrain Satellite Only with best Optuna params."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load best params
    optuna_dir = MODELS_DIR / "best_optuna_satellite"
    params_path = optuna_dir / "best_params.yaml"
    
    if not params_path.exists():
        logger.error(f"❌ Best params not found at {params_path}")
        logger.error("   Run: python -m solar_forecast.optuna_satellite_only")
        raise typer.Exit(1)
    
    with open(params_path) as f:
        best_params = yaml.safe_load(f)
    
    logger.info("=" * 70)
    logger.success(f"✓ Loaded best parameters")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)
    
    # Extract hyperparams
    lr = best_params.get("lr", 5e-4)
    weight_decay = best_params.get("weight_decay", 5e-4)
    batch_size = best_params.get("batch_size", 64)
    huber_delta = best_params.get("huber_delta", 1.0)
    max_grad_norm = best_params.get("max_grad_norm", 1.0)
    lr_patience = best_params.get("lr_patience", 7)
    lr_factor = best_params.get("lr_factor", 0.5)
    
    # Config
    base_cfg = load_model_cfg()
    cfg = build_config_from_params(base_cfg, best_params)
    
    # Data
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)
    (
        sat_tr, ground_tr, target_tr, ts_tr,
        sat_val, ground_val, target_val, ts_val,
    ) = split_by_random_days(sat, ground, target, timestamps, val_ratio, seed)
    
    train_loader = make_dataloader(
        sat_tr, ground_tr, target_tr, ts_tr,
        cfg=cfg, batch_size=batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        sat_val, ground_val, target_val, ts_val,
        cfg=cfg, batch_size=batch_size, shuffle=False
    )
    
    # Model
    model = SatelliteOnlyModel(
        cfg=cfg, cfg_sat=cfg["model"]["satellite"],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )
    
    # Train
    train_losses, val_losses = [], []
    best_val, best_epoch, wait = float("inf"), 0, 0
    
    logger.info(f"Training for {epochs} epochs...")
    logger.info("=" * 70)
    
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        logger.info(f"[{epoch:03d}/{epochs}] train={tr_loss:.4f} val={val_loss:.4f}")
        
        if val_loss < best_val:
            best_val, best_epoch, wait = val_loss, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "best_params": best_params,
                "epoch": epoch,
                "val_loss": val_loss,
            }, optuna_dir / "best_model.pt")
            logger.success(f"✓ New best (val={val_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                break
    
    logger.success(f"Best epoch: {best_epoch} | Val: {best_val:.4f}")
    
    # Save history
    pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }).to_csv(optuna_dir / "training_history.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best ({best_epoch})')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Satellite Only Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(optuna_dir / "training_curve.png", dpi=150)
    plt.close()
    
    with open(optuna_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)
    
    logger.success(f"✨ All saved to: {optuna_dir}")


if __name__ == "__main__":
    app()

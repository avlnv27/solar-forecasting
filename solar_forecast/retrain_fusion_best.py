"""
Retrain Fusion model with best hyperparameters from Optuna.

Automatically loads best params from models/best_optuna_fusion/best_params.yaml

Run:
python -m solar_forecast.retrain_fusion_best --epochs 200
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
from solar_forecast.nn.models.fusion import FusionModel
from solar_forecast.train import (
    load_model_cfg,
    prepare_data_and_graph,
    split_by_random_days,
    train_one_epoch,
    evaluate,
)

app = typer.Typer(help="Retrain Fusion with best Optuna hyperparameters")


def build_config_from_params(base_cfg: dict, params: dict) -> dict:
    """Apply best parameters to config."""
    import copy
    cfg = copy.deepcopy(base_cfg)

    # Ground
    if "ground_hidden_size" in params:
        cfg["model"]["ground"]["hidden_size"] = params["ground_hidden_size"]
        cfg["model"]["ground"]["output_size"] = params["ground_hidden_size"]
    if "ground_graph_layers" in params:
        cfg["model"]["ground"]["graph_layers"] = params["ground_graph_layers"]
    if "ground_time_layers" in params:
        cfg["model"]["ground"]["time_layers"] = params["ground_time_layers"]
    if "ground_activation" in params:
        cfg["model"]["ground"]["activation"] = params["ground_activation"]

    # Satellite
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

    # Fusion
    if "fusion_x1" in params:
        cfg["model"]["fusion"]["x1"] = params["fusion_x1"]
    if "fusion_x2" in params:
        cfg["model"]["fusion"]["x2"] = params["fusion_x2"]
    if "fusion_x3" in params:
        cfg["model"]["fusion"]["x3"] = params["fusion_x3"]
    if "fusion_dropout" in params:
        cfg["model"]["fusion"]["dropout"] = params["fusion_dropout"]

    return cfg


@app.command()
def retrain(
    epochs: int = 200,
    val_ratio: float = 0.2,
    patience: int = 25,
    seed: int = 42,
):
    """
    Retrain Fusion model with best Optuna hyperparameters.
    
    Loads parameters from models/best_optuna_fusion/best_params.yaml
    Saves results to models/best_optuna_fusion/
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load best parameters from Optuna
    optuna_dir = MODELS_DIR / "best_optuna_fusion"
    params_path = optuna_dir / "best_params.yaml"
    
    if not params_path.exists():
        logger.error(f"❌ Best params not found at {params_path}")
        logger.error("   Run Optuna search first: python -m solar_forecast.optuna_search")
        raise typer.Exit(1)
    
    with open(params_path) as f:
        best_params = yaml.safe_load(f)
    
    logger.info("=" * 70)
    logger.success(f"✓ Loaded best parameters from Optuna")
    logger.info("Best hyperparameters:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)
    
    # Extract training hyperparameters
    lr = best_params.get("lr", 5e-4)
    weight_decay = best_params.get("weight_decay", 5e-4)
    batch_size = best_params.get("batch_size", 64)
    huber_delta = best_params.get("huber_delta", 1.0)
    max_grad_norm = best_params.get("max_grad_norm", 1.0)
    lr_patience = best_params.get("lr_patience", 7)
    lr_factor = best_params.get("lr_factor", 0.5)
    
    # Load base config and apply best params
    base_cfg = load_model_cfg()
    cfg = build_config_from_params(base_cfg, best_params)
    
    # Prepare data
    logger.info("Preparing data...")
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
    
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # Create model
    logger.info("Creating model with best architecture...")
    model = FusionModel(
        cfg=cfg,
        cfg_sat=cfg["model"]["satellite"],
        cfg_ground=cfg["model"]["ground"],
        cfg_fusion=cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {n_params:,}")
    
    # Training setup
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )
    
    # Training loop
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    
    logger.info("=" * 70)
    logger.info(f"Starting training for {epochs} epochs (patience={patience})...")
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
            f"[{epoch:03d}/{epochs}] train={tr_loss:.4f} val={val_loss:.4f} lr={current_lr:.2e}"
        )
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            
            checkpoint_path = optuna_dir / "best_model.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
                "best_params": best_params,
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "best_val": best_val,
            }, checkpoint_path)
            
            logger.success(f"✓ New best model (val={val_loss:.4f})")
        else:
            wait += 1
            if wait >= patience:
                logger.warning(f"Early stopping at epoch {epoch}")
                break
    
    logger.info("=" * 70)
    logger.success(f"Training completed!")
    logger.success(f"Best epoch: {best_epoch} | Best val loss: {best_val:.4f}")
    logger.info("=" * 70)
    
    # Save training history
    history_path = optuna_dir / "training_history.csv"
    df_loss = pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
    })
    df_loss.to_csv(history_path, index=False)
    logger.success(f"Training history → {history_path}")
    
    # Save training curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_loss["epoch"], df_loss["train_loss"], label="Train", linewidth=2)
    plt.plot(df_loss["epoch"], df_loss["val_loss"], label="Validation", linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, 
                label=f'Best (epoch {best_epoch})')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Fusion Model Training (Best Optuna Params)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = optuna_dir / "training_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.success(f"Training plot → {plot_path}")
    
    # Save final config
    config_path = optuna_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    logger.success(f"Config → {config_path}")
    
    logger.info("=" * 70)
    logger.success(f"✨ All files saved to: {optuna_dir}")
    logger.info("=" * 70)
    logger.info("Files created:")
    logger.info(f"  • best_model.pt")
    logger.info(f"  • config.yaml")
    logger.info(f"  • training_history.csv")
    logger.info(f"  • training_curve.png")


if __name__ == "__main__":
    app()

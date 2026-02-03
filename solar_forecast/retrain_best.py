"""
Retrain FusionModel with best hyperparameters from Optuna study.

Usage:
    python -m solar_forecast.retrain_best --study-name fusion_hpo --storage "sqlite:///optuna_study.db"
"""

from pathlib import Path
import torch
from torch import nn
from loguru import logger
import typer
import yaml
import optuna

from solar_forecast.config.paths import PROCESSED_DATA_DIR, MODELS_DIR
from solar_forecast.nn.models.fusion import FusionModel
from solar_forecast.nn.utils import make_dataloader
from solar_forecast.train_huber_randomday import (
    load_model_cfg,
    prepare_data_and_graph,
    split_by_random_days,
    train_one_epoch,
    evaluate,
)
from solar_forecast.optuna_fusion import build_config_from_params

app = typer.Typer(help="Retrain with best Optuna hyperparameters")


@app.command()
def retrain(
    study_name: str = "fusion_hpo",
    storage: str = "sqlite:///optuna_study.db",
    val_ratio: float = 0.2,
    max_epochs: int = 200,
    patience: int = 25,
    seed: int = 42,
    checkpoint_dir: str = None,
):
    """Retrain model with best hyperparameters from Optuna study."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Optuna study
    logger.info(f"Loading study '{study_name}' from {storage}")
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    best_params = study.best_params
    best_value = study.best_value
    
    logger.success(f"Best trial: {study.best_trial.number}")
    logger.success(f"Best validation loss: {best_value:.6f}")
    logger.info("Best hyperparameters:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")
    
    # Load base config and apply best params
    base_cfg = load_model_cfg()
    cfg = build_config_from_params(base_cfg, best_params)
    
    # Extract training hyperparameters
    lr = best_params.get("lr", 1e-3)
    weight_decay = best_params.get("weight_decay", 1e-5)
    batch_size = best_params.get("batch_size", 64)
    huber_delta = best_params.get("huber_delta", 1.0)
    max_grad_norm = best_params.get("max_grad_norm", 1.0)
    lr_patience = best_params.get("lr_patience", 5)
    lr_factor = best_params.get("lr_factor", 0.5)
    
    logger.info("\nTraining hyperparameters:")
    logger.info(f"  lr: {lr}")
    logger.info(f"  weight_decay: {weight_decay}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  huber_delta: {huber_delta}")
    logger.info(f"  max_grad_norm: {max_grad_norm}")
    
    # Prepare data
    logger.info("\nPreparing data...")
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
    
    # Create model
    logger.info("\nCreating model...")
    model = FusionModel(
        cfg=cfg,
        cfg_sat=cfg["model"]["satellite"],
        cfg_ground=cfg["model"]["ground"],
        cfg_fusion=cfg["model"]["fusion"],
        A_ground=(edge_index.to(device), edge_weight.to(device)),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {n_params:,}")
    
    # Training setup
    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lr_factor, patience=lr_patience, min_lr=1e-6
    )
    
    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = MODELS_DIR / "best_optuna"
    else:
        checkpoint_dir = Path(checkpoint_dir)
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val = float("inf")
    wait = 0
    train_losses = []
    val_losses = []
    
    logger.info(f"\nStarting training for {max_epochs} epochs (patience={patience})...")
    logger.info("=" * 70)
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, edge_index, edge_weight, max_grad_norm
        )
        
        val_loss = evaluate(
            model, val_loader, criterion,
            device, edge_index, edge_weight
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_params': best_params,
                'config': cfg,
            }, checkpoint_path)
            
            logger.success(f"✓ New best model saved (val_loss: {val_loss:.6f})")
        else:
            wait += 1
            if wait >= patience:
                logger.warning(f"Early stopping triggered after {epoch} epochs")
                break
    
    logger.info("=" * 70)
    logger.success(f"Training completed!")
    logger.success(f"Best validation loss: {best_val:.6f}")
    logger.success(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")
    
    # Save training history
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    })
    history_path = checkpoint_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved to: {history_path}")
    
    # Save best params as YAML
    params_path = checkpoint_dir / "best_params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(best_params, f)
    logger.info(f"Best parameters saved to: {params_path}")
    
    # Save full config
    config_path = checkpoint_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    logger.info(f"Full config saved to: {config_path}")


if __name__ == "__main__":
    app()
"""
Hyperparameter optimization for SatelliteOnlyModel using Optuna.

UPDATED: Saves all outputs to models/best_optuna_satellite/

Run:
python -m solar_forecast.optuna_satellite_only --n-trials 75
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from loguru import logger
import typer
import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from solar_forecast.nn.utils import make_dataloader
from solar_forecast.config.paths import PROCESSED_DATA_DIR, MODEL_CONFIG, MODELS_DIR
from solar_forecast.nn.models.satellite_only import SatelliteOnlyModel
from solar_forecast.train_huber_randomday import (
    load_model_cfg,
    prepare_data_and_graph,
    split_by_random_days,
)

app = typer.Typer(help="Optuna hyperparameter optimization for SatelliteOnlyModel")


# ============================================================
# CONFIG BUILDER
# ============================================================

def build_config_from_params(base_cfg: dict, params: dict) -> dict:
    """Apply Optuna parameters to base config for Satellite Only."""
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


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch_satellite(
    model, dataloader, optimizer, criterion, device, max_grad_norm
) -> float:
    model.train()
    total_loss, n = 0.0, 0

    for batch in dataloader:
        optimizer.zero_grad()
        y_hat = model(
            batch["satellite"].to(device),
            x_ground=None, edge_index=None, edge_weight=None,
        )
        loss = criterion(y_hat, batch["target"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * len(batch["target"])
        n += len(batch["target"])

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_satellite(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss, n = 0.0, 0

    for batch in dataloader:
        y_hat = model(
            batch["satellite"].to(device),
            x_ground=None, edge_index=None, edge_weight=None,
        )
        loss = criterion(y_hat, batch["target"].to(device))
        total_loss += loss.item() * len(batch["target"])
        n += len(batch["target"])

    return total_loss / max(n, 1)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def create_objective(
    sat_tr, ground_tr, target_tr, ts_tr,
    sat_val, ground_val, target_val, ts_val,
    base_cfg, device, max_epochs=80, patience=12,
):

    def objective(trial: optuna.Trial) -> float:

        # Training Hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        huber_delta = trial.suggest_float("huber_delta", 0.1, 2.0)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0)
        lr_patience = trial.suggest_int("lr_patience", 3, 10)
        lr_factor = trial.suggest_float("lr_factor", 0.3, 0.7)

        # Satellite Architecture
        sat_feature_dim = trial.suggest_categorical("sat_feature_dim", [32, 64, 128])
        sat_temporal_layers = trial.suggest_int("sat_temporal_layers", 1, 3)
        sat_conv1_channels = trial.suggest_categorical("sat_conv1_channels", [16, 32, 64])
        sat_conv2_channels = trial.suggest_categorical("sat_conv2_channels", [32, 64, 128])
        sat_head_hidden = trial.suggest_categorical("sat_head_hidden_dim", [16, 32, 64])
        sat_use_attention = trial.suggest_categorical("sat_use_attention", [True, False])
        sat_attn_dim = trial.suggest_categorical("sat_attn_dim", [16, 32, 64])

        params = dict(
            sat_feature_dim=sat_feature_dim,
            sat_temporal_layers=sat_temporal_layers,
            sat_conv1_channels=sat_conv1_channels,
            sat_conv2_channels=sat_conv2_channels,
            sat_head_hidden_dim=sat_head_hidden,
            sat_use_attention=sat_use_attention,
            sat_attn_dim=sat_attn_dim,
        )

        cfg = build_config_from_params(base_cfg, params)

        train_loader = make_dataloader(
            sat_tr, ground_tr, target_tr, ts_tr,
            cfg=cfg, batch_size=batch_size, shuffle=True
        )
        val_loader = make_dataloader(
            sat_val, ground_val, target_val, ts_val,
            cfg=cfg, batch_size=batch_size, shuffle=False
        )

        try:
            model = SatelliteOnlyModel(
                cfg=cfg, cfg_sat=cfg["model"]["satellite"],
            ).to(device)
        except Exception as e:
            logger.warning(f"Model creation failed: {e}")
            raise optuna.TrialPruned()

        criterion = nn.HuberLoss(delta=huber_delta)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lr_patience, min_lr=1e-6
        )

        best_val = float("inf")
        wait = 0

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch_satellite(
                model, train_loader, optimizer, criterion, device, max_grad_norm
            )
            val_loss = evaluate_satellite(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return best_val

    return objective


# ============================================================
# CLI
# ============================================================

@app.command()
def search(
    n_trials: int = 75,
    val_ratio: float = 0.2,
    max_epochs: int = 80,
    patience: int = 12,
    seed: int = 42,
    study_name: str = "satellite_hpo",
    storage: Optional[str] = None,
    n_jobs: int = 1,
):
    """Run Optuna search for Satellite Only model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_model_cfg()
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    (
        sat_tr, ground_tr, target_tr, ts_tr,
        sat_val, ground_val, target_val, ts_val,
    ) = split_by_random_days(sat, ground, target, timestamps, val_ratio, seed)

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = create_objective(
        sat_tr, ground_tr, target_tr, ts_tr,
        sat_val, ground_val, target_val, ts_val,
        base_cfg=cfg, device=device,
        max_epochs=max_epochs, patience=patience,
    )

    logger.info(f"Starting Optuna search for Satellite Only with {n_trials} trials...")
    logger.info("=" * 70)
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    logger.info("=" * 70)
    logger.success("Optuna search completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.6f}")
    logger.info("Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # UPDATED: Save to models/best_optuna_satellite/
    output_dir = MODELS_DIR / "best_optuna_satellite"
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "best_params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(study.best_params, f)
    logger.success(f"Best params → {params_path}")

    trials_path = output_dir / "all_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    logger.success(f"All trials → {trials_path}")

    if len(study.trials) > 5:
        importance_path = output_dir / "param_importance.yaml"
        try:
            importance = optuna.importance.get_param_importances(study)
            with open(importance_path, "w") as f:
                yaml.dump(dict(importance), f)
            logger.success(f"Param importance → {importance_path}")
        except Exception as e:
            logger.warning(f"Could not compute importance: {e}")
    
    logger.info("=" * 70)
    logger.success(f"✨ All results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
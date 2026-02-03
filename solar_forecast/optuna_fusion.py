"""
Hyperparameter optimization for FusionModel using Optuna.

UPDATED: Saves all outputs to models/best_optuna_fusion/

Run:
python -m solar_forecast.optuna_fusion --n-trials 100
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
from solar_forecast.nn.models.fusion import FusionModel
from solar_forecast.train import (
    load_model_cfg,
    prepare_data_and_graph,
    split_by_random_days,
    train_one_epoch,
    evaluate,
)

app = typer.Typer(help="Optuna hyperparameter optimization for FusionModel")


# ============================================================
# CONFIG BUILDER
# ============================================================

def build_config_from_params(base_cfg: dict, params: dict) -> dict:
    """Apply Optuna parameters to base config."""
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

    # Fusion - FIXED: Use x1/x2/x3
    if "fusion_x1" in params:
        cfg["model"]["fusion"]["x1"] = params["fusion_x1"]
    
    if "fusion_x2" in params:
        cfg["model"]["fusion"]["x2"] = params["fusion_x2"]
    
    if "fusion_x3" in params:
        cfg["model"]["fusion"]["x3"] = params["fusion_x3"]
    
    if "fusion_dropout" in params:
        cfg["model"]["fusion"]["dropout"] = params["fusion_dropout"]

    return cfg


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def create_objective(
    sat_tr: torch.Tensor,
    ground_tr: torch.Tensor,
    target_tr: torch.Tensor,
    ts_tr: pd.DatetimeIndex,
    sat_val: torch.Tensor,
    ground_val: torch.Tensor,
    target_val: torch.Tensor,
    ts_val: pd.DatetimeIndex,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    base_cfg: dict,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 15,
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

        # Ground Model
        ground_hidden = trial.suggest_categorical("ground_hidden_size", [16, 32, 64])
        ground_graph_layers = trial.suggest_int("ground_graph_layers", 1, 3)
        ground_time_layers = trial.suggest_int("ground_time_layers", 1, 2)
        ground_activation = trial.suggest_categorical(
            "ground_activation", ["relu", "elu", "gelu", "selu"]
        )

        # Satellite Model
        sat_feature_dim = trial.suggest_categorical("sat_feature_dim", [32, 64, 128])
        sat_temporal_layers = trial.suggest_int("sat_temporal_layers", 1, 3)
        sat_conv1_channels = trial.suggest_categorical("sat_conv1_channels", [16, 32, 64])
        sat_conv2_channels = trial.suggest_categorical("sat_conv2_channels", [32, 64, 128])
        sat_head_hidden = trial.suggest_categorical("sat_head_hidden_dim", [16, 32, 64])

        # Fusion Model
        fusion_x1 = trial.suggest_categorical("fusion_x1", [32, 64, 128])
        fusion_x2 = trial.suggest_categorical("fusion_x2", [16, 32, 64])
        fusion_x3 = trial.suggest_categorical("fusion_x3", [8, 16, 32])
        fusion_dropout = trial.suggest_float("fusion_dropout", 0.1, 0.5)

        params = dict(
            ground_hidden_size=ground_hidden,
            ground_graph_layers=ground_graph_layers,
            ground_time_layers=ground_time_layers,
            ground_activation=ground_activation,
            sat_feature_dim=sat_feature_dim,
            sat_temporal_layers=sat_temporal_layers,
            sat_conv1_channels=sat_conv1_channels,
            sat_conv2_channels=sat_conv2_channels,
            sat_head_hidden_dim=sat_head_hidden,
            fusion_x1=fusion_x1,
            fusion_x2=fusion_x2,
            fusion_x3=fusion_x3,
            fusion_dropout=fusion_dropout,
        )

        cfg = build_config_from_params(base_cfg, params)

        # DataLoaders
        train_loader = make_dataloader(
            sat_tr, ground_tr, target_tr, ts_tr,
            cfg=cfg, batch_size=batch_size, shuffle=True
        )

        val_loader = make_dataloader(
            sat_val, ground_val, target_val, ts_val,
            cfg=cfg, batch_size=batch_size, shuffle=False
        )

        # Model
        try:
            model = FusionModel(
                cfg=cfg,
                cfg_sat=cfg["model"]["satellite"],
                cfg_ground=cfg["model"]["ground"],
                cfg_fusion=cfg["model"]["fusion"],
                A_ground=(edge_index.to(device), edge_weight.to(device)),
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
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, edge_index, edge_weight, max_grad_norm
            )

            val_loss = evaluate(
                model, val_loader, criterion,
                device, edge_index, edge_weight
            )

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
# CLI COMMANDS
# ============================================================

@app.command()
def search(
    n_trials: int = 100,
    val_ratio: float = 0.2,
    max_epochs: int = 100,
    patience: int = 15,
    seed: int = 42,
    study_name: str = "fusion_hpo",
    storage: Optional[str] = None,
    n_jobs: int = 1,
):
    """Run Optuna hyperparameter search for Fusion model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_model_cfg()
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    (
        sat_tr, ground_tr, target_tr, ts_tr,
        sat_val, ground_val, target_val, ts_val,
    ) = split_by_random_days(sat, ground, target, timestamps, val_ratio, seed)

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=5)

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
        edge_index, edge_weight,
        base_cfg=cfg,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
    )

    logger.info(f"Starting Optuna search with {n_trials} trials...")
    logger.info("=" * 70)
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    logger.info("=" * 70)
    logger.success("Optuna search completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.6f}")
    logger.info("Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # UPDATED: Save to models/best_optuna_fusion/
    output_dir = MODELS_DIR / "best_optuna_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best params
    params_path = output_dir / "best_params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(study.best_params, f)
    logger.success(f"Best params → {params_path}")

    # Save all trials
    trials_path = output_dir / "all_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    logger.success(f"All trials → {trials_path}")

    # Save parameter importance
    if len(study.trials) > 5:
        importance_path = output_dir / "param_importance.yaml"
        try:
            importance = optuna.importance.get_param_importances(study)
            with open(importance_path, "w") as f:
                yaml.dump(dict(importance), f)
            logger.success(f"Param importance → {importance_path}")
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
    
    logger.info("=" * 70)
    logger.success(f"✨ All results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
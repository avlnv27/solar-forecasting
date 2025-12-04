from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger
import typer
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from solar_forecast.nn.utils import (
    GroundPreprocessor,
    SatellitePreprocessor,
    make_dataloader,
)
from solar_forecast.config.paths import (
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    GROUND_DATASET_CONFIG, SATELLITE_DATASET_CONFIG,
    PREPROCESSING_DATASET_CONFIG, RAW_DATA_DIR
)
from solar_forecast.nn.models.fusion import FusionModel

"""
Run like this to test training:
python -m solar_forecast.main train-fusion \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --val-ratio 0.2
  
  """

app = typer.Typer(help="Preprocessing, debug forward and training for FusionModel.")


# Training utilities: backprop, eval, fit
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        sat = batch["satellite"].to(device)
        ground = batch["ground"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        y_pred = model(sat, ground, edge_index=edge_index, edge_weight=edge_weight)
        loss = criterion(y_pred, target)
        loss.backward()          
        optimizer.step()

        running_loss += loss.item() * target.size(0)
        n_samples += target.size(0)

    return running_loss / max(n_samples, 1)


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
    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        sat = batch["satellite"].to(device)
        ground = batch["ground"].to(device)
        target = batch["target"].to(device)

        y_pred = model(sat, ground, edge_index=edge_index, edge_weight=edge_weight)
        loss = criterion(y_pred, target)

        running_loss += loss.item() * target.size(0)
        n_samples += target.size(0)

    return running_loss / max(n_samples, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    n_epochs: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    ckpt_path: Optional[Path] = None,
) -> None:
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, edge_index, edge_weight
        )

        if val_loader is not None:
            val_loss = evaluate(
                model, val_loader, criterion, device, edge_index, edge_weight
            )
            logger.info(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )
        else:
            val_loss = float("nan")
            logger.info(
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"(no validation loader)"
            )

        # Save best model on validation loss
        if val_loader is not None and ckpt_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            logger.success(
                f"New best model saved to {ckpt_path} (val_loss={val_loss:.4f})"
            )


# Data preprocessing function for the pipeline
def prepare_data_and_graph(pp_cfg: dict) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor
]:
    """
    Pipeline:
      - preprocess ground
      - build graph
      - preprocess satellite
      - align
      - build tensors sat / ground / target

    Returns: 
      sat_tensor      (T, H, W)
      ground_tensor   (T, N) --> one feature per station
      target_tensor   (T,) --> irradiance to predict
      edge_index, edge_weight
    """
    start = pp_cfg["date_range"]["start"]
    end = pp_cfg["date_range"]["end"]
    freq = pd.to_timedelta(pp_cfg["frequency"])

    # GROUND PREPROCESSING
    gproc = GroundPreprocessor(cfg_path=GROUND_DATASET_CONFIG)

    gdf = gproc.load_and_merge()
    gdf = gdf.resample(freq).interpolate().ffill().bfill()
    gdf = gdf.loc[start:end]

    logger.success(f"Ground data: {gdf.shape}")

    # BUILDING GRAPH  
    logger.info("Building spatial graph from stations.")

    edge_index, edge_weight, coords = gproc.build_graph()

    edge_index = edge_index.to(torch.long)
    edge_weight = edge_weight.to(torch.float32)

    logger.success(
        f"Graph built -> edge_index={tuple(edge_index.shape)}, "
        f"edge_weight={tuple(edge_weight.shape)}"
    )

    # SATELLITE PREPROCESSING
    sproc = SatellitePreprocessor(
        cfg_path=SATELLITE_DATASET_CONFIG,
        raw_dir=RAW_DATA_DIR / "satellite",
        interim_dir=INTERIM_DATA_DIR / "satellite",
        processed_dir=PROCESSED_DATA_DIR / "satellite",
    )

    sdf = pd.read_csv(sproc.interim_dir / "satellite_irradiance.csv")
    sdf["time"] = pd.to_datetime(sdf["time"])

    sdf = (
        sdf.set_index("time")
        .groupby(["lat", "lon"])
        .resample(freq).mean()
        .interpolate()
        .ffill().bfill()
    )

    sdf.index = sdf.index.set_names(["lat_level", "lon_level", "time"])
    sdf = sdf.reset_index()
    sdf = sdf.drop(columns=["lat_level", "lon_level"])

    sdf = sdf[(sdf["time"] >= start) & (sdf["time"] <= end)]

    # ALIGN GROUND & SATELLITE 
    if not gdf.index.equals(sdf["time"]):
        logger.warning("Aligning timestamps...")
        common_idx = gdf.index.intersection(sdf["time"])
        gdf = gdf.loc[common_idx]
        sdf = sdf[sdf["time"].isin(common_idx)]
        logger.warning(f"Trimmed to {len(common_idx)} samples")

    logger.success("Ground & satellite aligned.")

    # SATELLITE TENSOR 
    sat_tensor = sproc.pivot_to_tensor(sdf)        # (T, H, W)
    logger.success(f"Satellite tensor: {sat_tensor.shape}")

    # GROUND & TARGET TENSORS
    ground_tensor = torch.tensor(gdf.values, dtype=torch.float32)            # (T, N)
    target_tensor = torch.tensor(gdf.iloc[:, 0].values, dtype=torch.float32) # (T,)

    logger.info(f"Ground tensor: {ground_tensor.shape}")
    logger.info(f"Target tensor: {target_tensor.shape}")

    return sat_tensor, ground_tensor, target_tensor, edge_index, edge_weight


# Normalisation (only on training set)
def normalize_time_series_tensors(
    sat_tensor: torch.Tensor,       # (T, H, W)
    ground_tensor: torch.Tensor,    # (T, N)
    target_tensor: torch.Tensor,    # (T,)
    n_train: int,
):
    """
    Calculates mean/std on training set (0:n_train) and applies normalisation on train + val (and later on test). 
    """

    eps = 1e-6

    # GROUND 
    ground_train = ground_tensor[:n_train]                     # (T_train, N)
    ground_mean = ground_train.mean(dim=0, keepdim=True)       # (1, N)
    ground_std = ground_train.std(dim=0, keepdim=True) + eps   # (1, N)

    ground_norm = (ground_tensor - ground_mean) / ground_std

    # TARGET 
    target_train = target_tensor[:n_train]                     # (T_train,)
    t_mean = target_train.mean()
    t_std = target_train.std() + eps

    target_norm = (target_tensor - t_mean) / t_std

    # ----- SATELLITE -----
    sat_train = sat_tensor[:n_train]                           # (T_train, H, W)
    sat_mean = sat_train.mean(dim=0, keepdim=True)             # (1, H, W)
    sat_std = sat_train.std(dim=0, keepdim=True) + eps         # (1, H, W)

    sat_norm = (sat_tensor - sat_mean) / sat_std

    logger.info("Normalization done.")
    return sat_norm, ground_norm, target_norm


@app.command()
def debug_forward(
    val_ratio: float = typer.Option(0.2, help="Fraction for validation when choosing train segment for normalization.")
):
    """
      - complete preprocessing
      - normalisation 
      - DataLoader
      - forward test of FusionModel
    """
    logger.info("Running full preprocessing + dataset + model debug forward")

    with open(PREPROCESSING_DATASET_CONFIG) as f:
        pp_cfg = yaml.safe_load(f)

    sat_tensor, ground_tensor, target_tensor, edge_index, edge_weight = prepare_data_and_graph(pp_cfg)

    # Training set choice
    n_timesteps = ground_tensor.shape[0]
    n_train = int(n_timesteps * (1.0 - val_ratio))
    logger.info(f"Using first {n_train}/{n_timesteps} timesteps for normalization in debug_forward.")

    sat_norm, ground_norm, target_norm = normalize_time_series_tensors(
        sat_tensor, ground_tensor, target_tensor, n_train=n_train
    )

    # DATASET + DATALOADER 
    loader = make_dataloader(
        sat_data=sat_norm,
        ground_data=ground_norm,
        target_data=target_norm,
        cfg=pp_cfg,
        batch_size=pp_cfg["batch_size"],
        shuffle=True,
    )

    batch = next(iter(loader))

    logger.info(f"Batch satellite: {batch['satellite'].shape}")
    logger.info(f"Batch ground:    {batch['ground'].shape}")
    logger.info(f"Batch target:    {batch['target'].shape}")

    # MODEL INSTANTIATION
    logger.info("Instantiating FusionModel with static graph...")

    model = FusionModel(
        cfg=pp_cfg,
        cfg_sat=pp_cfg["model"]["satellite"],
        cfg_ground=pp_cfg["model"]["ground"],
        cfg_fusion=pp_cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),
    )

    # TEST FORWARD PASS
    with torch.no_grad():
        y_pred = model(
            batch["satellite"],
            batch["ground"],
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    logger.info(f"Output y_pred: shape={y_pred.size()}")
    logger.info(f"Output sample: {y_pred[0]}")


@app.command()
def main():
    debug_forward()


@app.command()
def train_fusion(
    epochs: int = typer.Option(50, help="Number of training epochs."),
    batch_size: int = typer.Option(32, help="Batch size."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    weight_decay: float = typer.Option(1e-5, help="Weight decay for Adam."),
    val_ratio: float = typer.Option(0.2, help="Fraction of timesteps used for validation."),
    ckpt_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "checkpoints" / "fusion_best.pt",
        help="Path where to save the best model checkpoint.",
    ),
):
    """
    Training of FusionModel (CNN satellite + GNN ground + dense fusion) with back propagation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Starting training of FusionModel.")
    # CONFIG LOADING
    with open(PREPROCESSING_DATASET_CONFIG) as f:
        pp_cfg = yaml.safe_load(f)

    # PREPROCESS + GRAPH
    sat_tensor, ground_tensor, target_tensor, edge_index, edge_weight = prepare_data_and_graph(pp_cfg)

    # Split train / val
    n_timesteps = ground_tensor.shape[0]
    n_train = int(n_timesteps * (1.0 - val_ratio))
    n_val = n_timesteps - n_train

    if n_val <= 0:
        logger.warning("val_ratio too small or dataset too short --> no validation set.")
        n_train = n_timesteps
        n_val = 0

    logger.info(f"Splitting time series into {n_train} train steps and {n_val} val steps.")

    # Normalisation 
    sat_norm, ground_norm, target_norm = normalize_time_series_tensors(
        sat_tensor, ground_tensor, target_tensor, n_train=n_train
    )

    # Slicing after normalisation
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_timesteps) if n_val > 0 else slice(0, 0)

    sat_train = sat_norm[train_slice]
    ground_train = ground_norm[train_slice]
    target_train = target_norm[train_slice]

    if n_val > 0:
        sat_val = sat_norm[val_slice]
        ground_val = ground_norm[val_slice]
        target_val = target_norm[val_slice]
    else:
        sat_val = ground_val = target_val = None

    # DATALOADERS 
    train_loader = make_dataloader(
        sat_data=sat_train,
        ground_data=ground_train,
        target_data=target_train,
        cfg=pp_cfg,
        batch_size=batch_size,
        shuffle=True,
    )

    if n_val > 0:
        val_loader = make_dataloader(
            sat_data=sat_val,
            ground_data=ground_val,
            target_data=target_val,
            cfg=pp_cfg,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    # MODEL
    logger.info("Instantiating FusionModel with static graph for training...")
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = FusionModel(
        cfg=pp_cfg,
        cfg_sat=pp_cfg["model"]["satellite"],
        cfg_ground=pp_cfg["model"]["ground"],
        cfg_fusion=pp_cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),
    ).to(device)

    # LOSS & OPTIMIZER
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # TRAINING LOOP
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=epochs,
        edge_index=edge_index,
        edge_weight=edge_weight,
        ckpt_path=ckpt_path,
    )

    logger.success("Training complete.")


if __name__ == "__main__":
    app()

# solar_forecast/main.py
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
from solar_forecast.nn.models.fusion import FusionModel

from solar_forecast.nn.models.satellite_only import SatelliteOnlyModel
from solar_forecast.nn.models.ground_only import GroundOnlyModel

"""
Run:
python -m solar_forecast.main debug-forward
python -m solar_forecast.main train-fusion --epochs 50 --batch-size 32 --lr 1e-3 --val-ratio 0.2
"""

app = typer.Typer(help="Debug forward + training for FusionModel (using CLEANED processed data).")


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------
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
                f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )
        else:
            val_loss = float("nan")
            logger.info(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} (no val)")

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
            logger.success(f"New best model saved to {ckpt_path} (val_loss={val_loss:.4f})")


# ---------------------------------------------------------------------
# Data loading from FINAL CLEANED files (processed/)
# ---------------------------------------------------------------------
def _pivot_satellite_to_tensor(
    sdf: pd.DataFrame,
    time_col: str,
    lat_col: str,
    lon_col: str,
    value_col: str,
) -> torch.Tensor:
    """
    Convert satellite long dataframe into (T, H, W) tensor using a chosen value column.
    Assumes per-timestamp there is a fixed lat/lon grid.
    """
    need = {time_col, lat_col, lon_col, value_col}
    missing = need - set(sdf.columns)
    if missing:
        raise RuntimeError(f"Satellite clean CSV missing columns: {missing}")

    sdf = sdf.copy()
    sdf[time_col] = pd.to_datetime(sdf[time_col])

    times = pd.DatetimeIndex(sorted(sdf[time_col].unique()))
    lats = np.array(sorted(sdf[lat_col].unique()), dtype=float)
    lons = np.array(sorted(sdf[lon_col].unique()), dtype=float)

    H, W = len(lats), len(lons)
    if H == 0 or W == 0:
        raise RuntimeError("Satellite grid is empty (no unique lat/lon).")

    # map lat/lon to indices
    lat_to_i = {v: i for i, v in enumerate(lats)}
    lon_to_j = {v: j for j, v in enumerate(lons)}
    time_to_t = {t: k for k, t in enumerate(times)}

    arr = np.full((len(times), H, W), np.nan, dtype=np.float32)

    for row in sdf[[time_col, lat_col, lon_col, value_col]].itertuples(index=False):
        t, lat, lon, val = row
        ti = time_to_t[pd.Timestamp(t)]
        i = lat_to_i[float(lat)]
        j = lon_to_j[float(lon)]
        if val is not None:
            arr[ti, i, j] = float(val)

    # You probably want no NaNs after resampling/interpolation; but be safe:
    # Replace remaining NaNs by 0.0 (or you can raise).
    arr = np.nan_to_num(arr, nan=0.0)

    return torch.from_numpy(arr)


def prepare_data_and_graph(cfg: dict) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, pd.DatetimeIndex, torch.Tensor, torch.Tensor
]:
    start = pd.to_datetime(cfg["date_range"]["start"])
    end = pd.to_datetime(cfg["date_range"]["end"])

    # YAML-driven data config
    data_cfg = cfg.get("data", {})
    ground_value_col = data_cfg.get("ground_value_col", "irradiance")  # <-- now CSI by yaml
    target_station = data_cfg.get("target_station", None)

    sat_value_col = data_cfg.get("sat_value_col", None)  # MUST be set to CSI column in satellite clean
    sat_time_col = data_cfg.get("sat_time_col", "time")

    freq_str = cfg.get("frequency", "10min")
    _ = pd.to_timedelta(freq_str)

    # ---------------- GROUND (clean) ----------------
    gproc = GroundPreprocessor(cfg_path=GROUND_DATASET_CONFIG)
    ground_dir = PROCESSED_DATA_DIR / "ground"

    files = sorted(ground_dir.glob("*_clean.csv"))
    if not files:
        raise FileNotFoundError(f"No ground *_clean.csv found in {ground_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        time_col = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        if time_col is None or ground_value_col not in df.columns:
            continue

        df[time_col] = pd.to_datetime(df[time_col])

        if "station" in df.columns and len(df) > 0:
            st = str(df["station"].iloc[0]).upper()
        else:
            st = f.stem.split("_")[0].upper()

        sdf = df[[time_col, ground_value_col]].rename(columns={time_col: "time", ground_value_col: st})
        dfs.append(sdf)

    if not dfs:
        raise RuntimeError(
            f"No valid ground clean files with (time/timestamp, {ground_value_col}). "
            f"Check your YAML data.ground_value_col."
        )

    gdf = dfs[0]
    for d in dfs[1:]:
        gdf = gdf.merge(d, on="time", how="inner")

    gdf = gdf.sort_values("time")
    gdf = gdf[(gdf["time"] >= start) & (gdf["time"] <= end)]
    gdf = gdf.set_index("time")

    wanted = [s.upper() for s in gproc.test_stations]
    present = [c for c in wanted if c in gdf.columns]
    if len(present) == 0:
        raise RuntimeError("No ground stations matched YAML station list after merge.")
    gdf = gdf[present]

    logger.success(f"Ground clean data ({ground_value_col}): {gdf.shape}")

    # ---------------- GRAPH ----------------
    logger.info("Building spatial graph from stations...")
    edge_index, edge_weight, _coords = gproc.build_graph()
    edge_index = edge_index.to(torch.long)
    edge_weight = edge_weight.to(torch.float32)
    logger.success(
        f"Graph built -> edge_index={tuple(edge_index.shape)}, edge_weight={tuple(edge_weight.shape)}"
    )

    # ---------------- SATELLITE (clean) ----------------
    sproc = SatellitePreprocessor(
        cfg_path=SATELLITE_DATASET_CONFIG,
        raw_dir=RAW_DATA_DIR / "satellite",
        interim_dir=INTERIM_DATA_DIR / "satellite",
        processed_dir=PROCESSED_DATA_DIR / "satellite",
    )

    sat_csv = PROCESSED_DATA_DIR / "satellite" / "satellite_irradiance_clean.csv"
    if not sat_csv.exists():
        raise FileNotFoundError(f"Missing satellite clean CSV: {sat_csv}")

    sdf = pd.read_csv(sat_csv)
    if sat_time_col not in sdf.columns:
        raise RuntimeError(f"Satellite clean CSV missing '{sat_time_col}' column.")
    sdf[sat_time_col] = pd.to_datetime(sdf[sat_time_col])
    sdf = sdf[(sdf[sat_time_col] >= start) & (sdf[sat_time_col] <= end)]
    sdf = sdf.sort_values(sat_time_col)

    # Ensure YAML specifies which satellite value column to use
    if sat_value_col is None:
        raise RuntimeError(
            "You must set cfg['data']['sat_value_col'] to the CSI column name in satellite_irradiance_clean.csv"
        )

    # ---------------- ALIGN ----------------
    sat_times = pd.DatetimeIndex(sdf[sat_time_col].unique())
    common_idx = gdf.index.intersection(sat_times).sort_values()
    if len(common_idx) == 0:
        raise RuntimeError("No common timestamps between ground and satellite cleaned data.")

    gdf = gdf.loc[common_idx]
    sdf = sdf[sdf[sat_time_col].isin(common_idx)].sort_values(sat_time_col)
    logger.success(f"Ground & satellite aligned: {len(common_idx)} samples")

    # ---------------- TENSORS ----------------
    # Satellite: YAML-driven pivot on CSI column
    sat_tensor = _pivot_satellite_to_tensor(
        sdf=sdf,
        time_col=sat_time_col,
        lat_col=sproc.col_lat,
        lon_col=sproc.col_lon,
        value_col=sat_value_col,
    )

    # Ground: YAML-driven value (CSI)
    ground_tensor = torch.tensor(gdf.values, dtype=torch.float32)

    # Target: YAML-driven station selection (still predicting one station)
    if target_station is not None:
        target_station = str(target_station).upper()
        if target_station not in gdf.columns:
            raise RuntimeError(f"target_station={target_station} not found in ground columns: {list(gdf.columns)}")
        target_series = gdf[target_station]
        logger.info(f"Target station: {target_station}")
    else:
        target_series = gdf.iloc[:, 0]
        logger.info(f"Target station: {gdf.columns[0]} (default first)")

    target_tensor = torch.tensor(target_series.values, dtype=torch.float32)
    timestamps = gdf.index

    logger.info(f"Satellite tensor: {sat_tensor.shape}")
    logger.info(f"Ground tensor:    {ground_tensor.shape}")
    logger.info(f"Target tensor:    {target_tensor.shape}")

    return sat_tensor, ground_tensor, target_tensor, timestamps, edge_index, edge_weight


def load_model_cfg() -> dict:
    """Loads the YAML you want to call 'model' config."""
    with open(MODEL_CONFIG) as f:
        cfg = yaml.safe_load(f)

    # sanity checks
    required = ["date_range", "past_timesteps", "future_timesteps", "model"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"MODEL_CONFIG missing keys: {missing}")

    return cfg


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------
@app.command()
def debug_forward(
    val_ratio: float = typer.Option(0.2, help="Used only to pick a train segment for quick sanity."),
):
    """
    Loads cleaned data -> DataLoader (same-day windows enforced inside Dataset) -> forward pass.
    """
    logger.info("Debug forward using CLEANED processed data.")
    cfg = load_model_cfg()

    sat_tensor, ground_tensor, target_tensor, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    loader = make_dataloader(
        sat_data=sat_tensor,
        ground_data=ground_tensor,
        target_data=target_tensor,
        timestamps=timestamps,
        cfg=cfg,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
    )

    batch = next(iter(loader))
    logger.info(f"Batch satellite: {batch['satellite'].shape}")
    logger.info(f"Batch ground:    {batch['ground'].shape}")
    logger.info(f"Batch target:    {batch['target'].shape}")

    model = FusionModel(
        cfg=cfg,
        cfg_sat=cfg["model"]["satellite"],
        cfg_ground=cfg["model"]["ground"],
        cfg_fusion=cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),
    )

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
    Training of FusionModel (CNN satellite + GNN ground + dense fusion) with backprop.
    Uses CLEANED processed data + same-day windowing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_model_cfg()

    sat_tensor, ground_tensor, target_tensor, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    # Split train/val along the aligned timeline
    n_timesteps = ground_tensor.shape[0]
    n_train = int(n_timesteps * (1.0 - val_ratio))
    n_val = n_timesteps - n_train

    if n_val <= 0:
        logger.warning("No validation set (val_ratio too small or dataset too short).")
        n_train = n_timesteps
        n_val = 0

    logger.info(f"Split: {n_train} train steps, {n_val} val steps.")

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_timesteps) if n_val > 0 else slice(0, 0)

    sat_train = sat_tensor[train_slice]
    ground_train = ground_tensor[train_slice]
    target_train = target_tensor[train_slice]
    ts_train = timestamps[train_slice]

    if n_val > 0:
        sat_val = sat_tensor[val_slice]
        ground_val = ground_tensor[val_slice]
        target_val = target_tensor[val_slice]
        ts_val = timestamps[val_slice]
    else:
        sat_val = ground_val = target_val = ts_val = None

    train_loader = make_dataloader(
        sat_data=sat_train,
        ground_data=ground_train,
        target_data=target_train,
        timestamps=ts_train,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=True,
    )

    if n_val > 0:
        val_loader = make_dataloader(
            sat_data=sat_val,
            ground_data=ground_val,
            target_data=target_val,
            timestamps=ts_val,
            cfg=cfg,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        val_loader = None

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = FusionModel(
        cfg=cfg,
        cfg_sat=cfg["model"]["satellite"],
        cfg_ground=cfg["model"]["ground"],
        cfg_fusion=cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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




@app.command()
def train_satellite(
    epochs: int = typer.Option(50),
    batch_size: int = typer.Option(32),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-5),
    val_ratio: float = typer.Option(0.2),
    ckpt_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "checkpoints" / "satellite_only_best.pt"
    ),
):
    """
    Train satellite-only CNN baseline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[SAT] Using device: {device}")

    cfg = load_model_cfg()

    sat_tensor, ground_tensor, target_tensor, timestamps, edge_index, edge_weight = (
        prepare_data_and_graph(cfg)
    )

    # --- same chronological split ---
    n = len(target_tensor)
    n_train = int(n * (1.0 - val_ratio))

    sat_train = sat_tensor[:n_train]
    sat_val   = sat_tensor[n_train:]

    target_train = target_tensor[:n_train]
    target_val   = target_tensor[n_train:]

    ts_train = timestamps[:n_train]
    ts_val   = timestamps[n_train:]

    train_loader = make_dataloader(
        sat_data=sat_train,
        ground_data=torch.zeros_like(ground_tensor[:n_train]),  # dummy
        target_data=target_train,
        timestamps=ts_train,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_dataloader(
        sat_data=sat_val,
        ground_data=torch.zeros_like(ground_tensor[n_train:]),  # dummy
        target_data=target_val,
        timestamps=ts_val,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=False,
    )

    model = SatelliteOnlyModel(
        cfg=cfg,
        cfg_sat=cfg["model"]["satellite"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=epochs,
        edge_index=None,
        edge_weight=None,
        ckpt_path=ckpt_path,
    )

    logger.success("Satellite-only training complete.")




@app.command()
def train_ground(
    epochs: int = typer.Option(50),
    batch_size: int = typer.Option(32),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-5),
    val_ratio: float = typer.Option(0.2),
    ckpt_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "checkpoints" / "ground_only_best.pt"
    ),
):
    """
    Train ground-only GNN baseline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[GNN] Using device: {device}")

    cfg = load_model_cfg()

    sat_tensor, ground_tensor, target_tensor, timestamps, edge_index, edge_weight = (
        prepare_data_and_graph(cfg)
    )

    # --- same chronological split ---
    n = len(target_tensor)
    n_train = int(n * (1.0 - val_ratio))

    ground_train = ground_tensor[:n_train]
    ground_val   = ground_tensor[n_train:]

    target_train = target_tensor[:n_train]
    target_val   = target_tensor[n_train:]

    ts_train = timestamps[:n_train]
    ts_val   = timestamps[n_train:]

    train_loader = make_dataloader(
        sat_data=torch.zeros_like(sat_tensor[:n_train]),  # dummy
        ground_data=ground_train,
        target_data=target_train,
        timestamps=ts_train,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_dataloader(
        sat_data=torch.zeros_like(sat_tensor[n_train:]),  # dummy
        ground_data=ground_val,
        target_data=target_val,
        timestamps=ts_val,
        cfg=cfg,
        batch_size=batch_size,
        shuffle=False,
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = GroundOnlyModel(
        cfg=cfg,
        cfg_ground=cfg["model"]["ground"],
        A_ground=(edge_index, edge_weight),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

    logger.success("Ground-only training complete.")



@app.command()
def main():
    debug_forward()


if __name__ == "__main__":
    app()




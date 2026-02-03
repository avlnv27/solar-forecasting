# solar_forecast/data/load_data.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from solar_forecast.nn.utils import (
    GroundPreprocessor,
    SatellitePreprocessor,
)
from solar_forecast.config.paths import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    GROUND_DATASET_CONFIG,
    SATELLITE_DATASET_CONFIG,
    MODEL_CONFIG,
)


# ============================================================
# CONFIG LOADER
# ============================================================

def load_model_cfg() -> dict:
    """Load main model YAML config."""
    with open(MODEL_CONFIG) as f:
        cfg = yaml.safe_load(f)

    required = ["date_range", "past_timesteps", "future_timesteps", "model"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"MODEL_CONFIG missing keys: {missing}")

    return cfg


# ============================================================
# SATELLITE PIVOT
# ============================================================

def pivot_satellite_to_tensor(
    sdf: pd.DataFrame,
    time_col: str,
    lat_col: str,
    lon_col: str,
    value_col: str,
) -> torch.Tensor:
    """
    Convert long satellite dataframe into (T, H, W) tensor.
    """
    sdf = sdf.copy()
    sdf[time_col] = pd.to_datetime(sdf[time_col])

    times = pd.DatetimeIndex(sorted(sdf[time_col].unique()))
    lats = np.array(sorted(sdf[lat_col].unique()), dtype=float)
    lons = np.array(sorted(sdf[lon_col].unique()), dtype=float)

    H, W = len(lats), len(lons)
    if H == 0 or W == 0:
        raise RuntimeError("Satellite grid is empty.")

    lat_to_i = {v: i for i, v in enumerate(lats)}
    lon_to_j = {v: j for j, v in enumerate(lons)}
    time_to_t = {t: k for k, t in enumerate(times)}

    arr = np.zeros((len(times), H, W), dtype=np.float32)

    for t, lat, lon, val in sdf[[time_col, lat_col, lon_col, value_col]].itertuples(index=False):
        ti = time_to_t[pd.Timestamp(t)]
        i = lat_to_i[float(lat)]
        j = lon_to_j[float(lon)]
        if val is not None:
            arr[ti, i, j] = float(val)

    return torch.from_numpy(arr)


# ============================================================
# MAIN DATA LOADER
# ============================================================

def prepare_data_and_graph(
    cfg: dict,
) -> Tuple[
    torch.Tensor,        # satellite tensor (T, H, W)
    torch.Tensor,        # ground tensor (T, N)
    torch.Tensor,        # target tensor (T,)
    pd.DatetimeIndex,    # timestamps
    torch.Tensor,        # edge_index
    torch.Tensor,        # edge_weight
]:
    """
    Load CLEANED data, align satellite & ground, build graph, return tensors.
    """

    start = pd.to_datetime(cfg["date_range"]["start"])
    end = pd.to_datetime(cfg["date_range"]["end"])

    data_cfg = cfg.get("data", {})
    ground_value_col = data_cfg.get("ground_value_col", "clear_sky_index")
    sat_value_col = data_cfg.get("sat_value_col")
    sat_time_col = data_cfg.get("sat_time_col", "time")
    target_station = data_cfg.get("target_station", None)

    if sat_value_col is None:
        raise RuntimeError("cfg['data']['sat_value_col'] must be set (CSI column).")

    # ========================================================
    # GROUND (CLEAN)
    # ========================================================
    gproc = GroundPreprocessor(cfg_path=GROUND_DATASET_CONFIG)
    ground_dir = PROCESSED_DATA_DIR / "ground"

    files = sorted(ground_dir.glob("*_clean.csv"))
    if not files:
        raise FileNotFoundError(f"No ground *_clean.csv in {ground_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        time_col = "time" if "time" in df.columns else "timestamp"
        if ground_value_col not in df.columns:
            continue

        df[time_col] = pd.to_datetime(df[time_col])
        station = f.stem.split("_")[0].upper()
        sdf = df[[time_col, ground_value_col]].rename(
            columns={time_col: "time", ground_value_col: station}
        )
        dfs.append(sdf)

    if not dfs:
        raise RuntimeError("No valid ground clean files after filtering.")

    gdf = dfs[0]
    for d in dfs[1:]:
        gdf = gdf.merge(d, on="time", how="inner")

    gdf = gdf.sort_values("time")
    gdf = gdf[(gdf["time"] >= start) & (gdf["time"] <= end)]
    gdf = gdf.set_index("time")

    stations = [s.upper() for s in gproc.test_stations]
    gdf = gdf[[c for c in stations if c in gdf.columns]]

    logger.success(f"Ground tensor shape: {gdf.shape}")

    # ========================================================
    # GRAPH
    # ========================================================
    edge_index, edge_weight, _ = gproc.build_graph()
    edge_index = edge_index.to(torch.long)
    edge_weight = edge_weight.to(torch.float32)

    # ========================================================
    # SATELLITE (CLEAN)
    # ========================================================
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
    sdf[sat_time_col] = pd.to_datetime(sdf[sat_time_col])
    sdf = sdf[(sdf[sat_time_col] >= start) & (sdf[sat_time_col] <= end)]

    # ========================================================
    # ALIGN
    # ========================================================
    sat_times = pd.DatetimeIndex(sdf[sat_time_col].unique())
    common_times = gdf.index.intersection(sat_times).sort_values()

    if len(common_times) == 0:
        raise RuntimeError("No common timestamps between ground and satellite.")

    gdf = gdf.loc[common_times]
    sdf = sdf[sdf[sat_time_col].isin(common_times)]

    # ========================================================
    # TENSORS
    # ========================================================
    sat_tensor = pivot_satellite_to_tensor(
        sdf=sdf,
        time_col=sat_time_col,
        lat_col=sproc.col_lat,
        lon_col=sproc.col_lon,
        value_col=sat_value_col,
    )

    ground_tensor = torch.tensor(gdf.values, dtype=torch.float32)

    if target_station is not None:
        target_station = target_station.upper()
        if target_station not in gdf.columns:
            raise RuntimeError(f"Target station {target_station} not found.")
        target = gdf[target_station]
    else:
        target = gdf.iloc[:, 0]

    target_tensor = torch.tensor(target.values, dtype=torch.float32)
    timestamps = gdf.index

    logger.success(f"Satellite tensor: {sat_tensor.shape}")
    logger.success(f"Ground tensor:    {ground_tensor.shape}")
    logger.success(f"Target tensor:    {target_tensor.shape}")

    return (
        sat_tensor,
        ground_tensor,
        target_tensor,
        timestamps,
        edge_index,
        edge_weight,
    )

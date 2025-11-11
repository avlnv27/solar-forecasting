from pathlib import Path
import pandas as pd
from loguru import logger
import typer
import yaml
import torch

# Local imports
from solar_forecast.datasets.ground_preprocessing import GroundPreprocessor
from solar_forecast.datasets.satellite_preprocessing import SatellitePreprocessor
from solar_forecast.config.paths import (
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    GROUND_DATASET_CONFIG, SATELLITE_DATASET_CONFIG,
    PREPROCESSING_DATASET_CONFIG, RAW_DATA_DIR
)
from solar_forecast.nn.models.fusion import FusionModel


app = typer.Typer()


@app.command()
def main():
    logger.info("Running full preprocessing pipeline (ground + satellite)")

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    with open(PREPROCESSING_DATASET_CONFIG) as f:
        pp_cfg = yaml.safe_load(f)
    with open(GROUND_DATASET_CONFIG) as f:
        ground_cfg = yaml.safe_load(f)
    with open(SATELLITE_DATASET_CONFIG) as f:
        sat_cfg = yaml.safe_load(f)

    start, end = pp_cfg["date_range"]["start"], pp_cfg["date_range"]["end"]
    freq = pd.to_timedelta(pp_cfg.get("frequency", "10min"))

    # Ground data
    gproc = GroundPreprocessor(
        cfg_path=GROUND_DATASET_CONFIG,
        raw_dir=RAW_DATA_DIR / "ground",
        interim_dir=INTERIM_DATA_DIR / "ground",
        processed_dir=PROCESSED_DATA_DIR / "ground",
    )

    gdf = gproc.load_and_merge()
    gdf = gdf.resample(freq).interpolate().ffill().bfill()
    gdf = gdf.loc[start:end]
    logger.info(f"Ground data resampled to {freq}, range {start}–{end}")
    print(gdf.head())

    # Satellite data
    sproc = SatellitePreprocessor(
        cfg_path=SATELLITE_DATASET_CONFIG,
        raw_dir=RAW_DATA_DIR / "satellite",
        interim_dir=INTERIM_DATA_DIR / "satellite",
        processed_dir=PROCESSED_DATA_DIR / "satellite",
    )
    sdf = sproc.redownload_csv()
    sdf = sproc.pivot_to_matrix(sdf)
    sdf = sdf.resample(freq).interpolate().ffill().bfill()
    sdf = sdf.loc[start:end]
    logger.info(f"Ground data resampled to {freq}, range {start}–{end}")
    print(sdf.head())

    # Align indices
    if not gdf.index.equals(sdf.index):
        common_index = gdf.index.intersection(sdf.index)
        gdf, sdf = gdf.loc[common_index], sdf.loc[common_index]
        logger.warning(f"Ground & satellite time indices misaligned — trimmed to {len(common_index)} samples.")

    logger.success("All data processed and aligned at 10min frequency — ready for SpatioTemporalDataset!")

    # Convert to tensors
    # Ground → (B, T, N, F)
    X_ground = torch.tensor(gdf.values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    logger.info(f"Ground tensor shape: {tuple(X_ground.shape)}")

    # Satellite → (B, C, H, W)
    side = int(len(sdf.columns) ** 0.5)
    # Take the mean image over time (you can also pick sdf.iloc[0] for one frame)
    X_sat = torch.tensor(sdf.mean(axis=0).values, dtype=torch.float32).view(1, 1, side, side)
    logger.info(f"Satellite tensor shape: {tuple(X_sat.shape)}")

    # Fusion model
    model = FusionModel(
        cfg_sat=sat_cfg["model"],
        cfg_ground=ground_cfg["model"],
        n_ground_nodes=gdf.shape[1],
        A_ground=None
    )

    # # ------------------------------------------------------------------
    # # Forward pass test
    # # ------------------------------------------------------------------
    # with torch.no_grad():
    #     y_pred = model(X_sat, X_ground)
    #     logger.success(f"✅ Forward pass OK — output shape: {tuple(y_pred.shape)}")


if __name__ == "__main__":
    app()

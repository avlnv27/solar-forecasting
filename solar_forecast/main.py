from pathlib import Path
import pandas as pd
from loguru import logger
import typer
import yaml
import torch

# ---- Correct imports ----
from solar_forecast.nn.utils import (
    GroundPreprocessor,
    SatellitePreprocessor,
    ForecastDataset,
    make_dataloader,
)
from solar_forecast.config.paths import (
    INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
    GROUND_DATASET_CONFIG, SATELLITE_DATASET_CONFIG,
    PREPROCESSING_DATASET_CONFIG, RAW_DATA_DIR
)
from solar_forecast.nn.models.fusion import FusionModel


app = typer.Typer()


@app.command()
def main():
    logger.info("Running full preprocessing + dataset + model test")

    # ---------------- CONFIG LOADING ----------------
    with open(PREPROCESSING_DATASET_CONFIG) as f:
        pp_cfg = yaml.safe_load(f)

    past_steps = pp_cfg["past_timesteps"]
    future_steps = pp_cfg["future_timesteps"]

    start = pp_cfg["date_range"]["start"]
    end = pp_cfg["date_range"]["end"]
    freq = pd.to_timedelta(pp_cfg["frequency"])

    # ---------------- GROUND PREPROCESSING ----------------
    gproc = GroundPreprocessor(cfg_path=GROUND_DATASET_CONFIG)

    gdf = gproc.load_and_merge()
    gdf = gdf.resample(freq).interpolate().ffill().bfill()
    gdf = gdf.loc[start:end]

    logger.success(f"Ground data: {gdf.shape}")

    # ---------------- GRAPH BUILD ----------------
    logger.info("Building spatial graph from stations...")

    edge_index, edge_weight, coords = gproc.build_graph()

    # Put graph tensors on consistent dtypes
    edge_index = edge_index.to(torch.long)
    edge_weight = edge_weight.to(torch.float32)

    logger.success(
        f"Graph built → edge_index={tuple(edge_index.shape)}, "
        f"edge_weight={tuple(edge_weight.shape)}"
    )

    # ---------------- SATELLITE PREPROCESS ----------------
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

    # ---- ALIGN GROUND & SATELLITE ----
    if not gdf.index.equals(sdf["time"]):
        logger.warning("Aligning timestamps...")
        common_idx = gdf.index.intersection(sdf["time"])
        gdf = gdf.loc[common_idx]
        sdf = sdf[sdf["time"].isin(common_idx)]
        logger.warning(f"Trimmed to {len(common_idx)} samples")

    logger.success("Ground & satellite aligned.")

    # ---------------- SATELLITE TENSOR ----------------
    sat_tensor = sproc.pivot_to_tensor(sdf)
    logger.success(f"Satellite tensor: {sat_tensor.shape}")

    # ---------------- GROUND TENSOR ----------------
    # ---- Ground tensor (T, N, 1) → expected by ForecastDataset ----
    ground_tensor = torch.tensor(gdf.values, dtype=torch.float32)
    # shape now: (T, N, 1)

    # ---- Target tensor (T,) ----
    target_tensor = torch.tensor(gdf.iloc[:, 0].values, dtype=torch.float32)

    logger.info(f"Ground tensor: {ground_tensor.shape}")
    logger.info(f"Target tensor: {target_tensor.shape}")

    # ---------------- DATASET + DATALOADER ----------------
    loader = make_dataloader(
        sat_data=sat_tensor,
        ground_data=ground_tensor,
        target_data=target_tensor,
        cfg=pp_cfg,
        batch_size=pp_cfg["batch_size"],
        shuffle=True
    )

    batch = next(iter(loader))

    logger.info(f"Batch satellite: {batch['satellite'].shape}")
    logger.info(f"Batch ground:    {batch['ground'].shape}")
    logger.info(f"Batch target:    {batch['target'].shape}")

    # ---------------- MODEL INSTANTIATION ----------------
    logger.info("Instantiating FusionModel with static graph...")

    model = FusionModel(
        cfg=pp_cfg,
        cfg_sat=pp_cfg["model"]["satellite"],
        cfg_ground=pp_cfg["model"]["ground"],
        cfg_fusion=pp_cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),   # PASS THE GRAPH HERE
    )

    # ---------------- TEST FORWARD PASS ----------------
    with torch.no_grad():
        y_pred = model(
            batch["satellite"],
            batch["ground"],
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        # print(y_pred.shape)

    logger.info(f"Batch satellite: {y_pred.size()}")
    logger.info(f"Batch satellite: {y_pred}")


    # logger.success(f"Forward OK — output shape: {tuple(y_pred.shape)}")



if __name__ == "__main__":
    app()

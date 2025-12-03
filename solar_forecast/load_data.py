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
    GROUND_DATASET_CONFIG, SATELLITE_DATASET_CONFIG, RAW_DATA_DIR
)


app = typer.Typer()


@app.command()
def main():
    logger.info("Preprocessing ground + satellite data")

    # Ground data
    gproc = GroundPreprocessor(cfg_path=GROUND_DATASET_CONFIG)

    stations_meta = gproc.load_station_metadata()
    raw_paths = gproc.download_all_stations()
    processed = [gproc.process_station_file(p) for p in raw_paths if p is not None]
    gproc.generate_summary(processed, stations_meta)

    # Satellite data
    sproc = SatellitePreprocessor(
        cfg_path=SATELLITE_DATASET_CONFIG,
        raw_dir=RAW_DATA_DIR / "satellite",
        interim_dir=INTERIM_DATA_DIR / "satellite",
        processed_dir=PROCESSED_DATA_DIR / "satellite",
    )

    logger.info("Processing satellite data...")
    sdf = sproc.load_from_nc()

    logger.success(f"Satellite data ready ({sdf.shape[0]} timesteps × {sdf.shape[1]} pixels)")


if __name__ == "__main__":
    app()

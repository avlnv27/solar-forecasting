from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm
import typer

from solar_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def process_ground(
    input_path: Path = RAW_DATA_DIR / "ogd-smn_puy_t_historical_2020-2029.csv",
    output_path: Path = PROCESSED_DATA_DIR / "ground_processed.csv",
):
    """
    Load raw ground station data, clean, and save processed version.
    """

    logger.info(f"Loading raw ground data from {input_path}")
    df = pd.read_csv(input_path)

    # Cleaning steps
    logger.info("Cleaning ground data...") 
    # 1. Parse timestamps
    if "reference_timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["reference_timestamp"])

    # 2. Sort by time 
    df = df.sort_values("timestamp")

    #3. Rename gre000z0 column to irradiance
    if "gre000z0" in df.columns:
        df = df.rename(columns={"gre000z0": "irradiance"})
    
    # 3. Fill missing values (basic example: forward fill)
    df = df.fillna(method="ffill")

    # 4. Optionally: restrict to subset of columns
    keep_cols = ["timestamp", "station_abbr", "irradiance"]
    df = df[keep_cols]

    # -----------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.success(f"Processed ground data saved to {output_path}")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

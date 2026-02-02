from loguru import logger
import typer
from pathlib import Path
import yaml

from solar_forecast.nn.utils import (
    GroundPreprocessor,
    SatellitePreprocessor,
)
from solar_forecast.config.paths import (
    GROUND_DATASET_CONFIG,
    SATELLITE_DATASET_CONFIG,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
python -m solar_forecast.preprocess_data
"""

app = typer.Typer()


# ---------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------
def load_csi_and_time_cols(cfg_path: Path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    csi_col = cfg["csi"]["csi"]
    time_col = cfg["columns"]["renamed"]["timestamp"]

    return csi_col, time_col


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_csi_histogram_cleaned(
    ground_dir: Path,
    satellite_csv: Path,
    station_code: str = "PUY",
    bins: int = 60,
    csi_range: tuple[float, float] = (0.0, 1.5),
    out_path: Path | None = None,
):
    """
    Histogram of CSI distributions on CLEANED data.
    Column names are read dynamically from YAML configs.
    """

    # --- Load column names from YAML ---
    ground_csi_col, ground_time_col = load_csi_and_time_cols(GROUND_DATASET_CONFIG)
    sat_csi_col, sat_time_col = load_csi_and_time_cols(SATELLITE_DATASET_CONFIG)

    # ---------------- Ground ----------------
    station_code = station_code.lower()
    g_files = list(ground_dir.glob(f"*{station_code}*_clean.csv"))
    if not g_files:
        logger.warning(f"[PLOT] No cleaned ground file found for {station_code.upper()}")
        return

    gdf = pd.read_csv(g_files[0])
    if ground_csi_col not in gdf.columns:
        logger.warning(f"[PLOT] Missing {ground_csi_col} in ground file")
        return

    g_csi = (
        gdf[ground_csi_col]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # ---------------- Satellite ----------------
    if not satellite_csv.exists():
        logger.warning(f"[PLOT] Satellite CSV not found: {satellite_csv}")
        return

    sdf = pd.read_csv(satellite_csv)
    if sat_time_col not in sdf.columns or sat_csi_col not in sdf.columns:
        logger.warning(
            f"[PLOT] Satellite CSV missing required columns: "
            f"{sat_time_col}, {sat_csi_col}"
        )
        return

    sdf[sat_time_col] = pd.to_datetime(sdf[sat_time_col])
    s_csi = (
        sdf.groupby(sat_time_col)[sat_csi_col]
        .median()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # ---------------- Plot ----------------
    plt.figure(figsize=(8, 5))

    plt.hist(
        g_csi,
        bins=bins,
        range=csi_range,
        density=True,
        alpha=0.6,
        label=f"Ground CSI ({station_code.upper()})",
    )
    plt.hist(
        s_csi,
        bins=bins,
        range=csi_range,
        density=True,
        alpha=0.6,
        label="Satellite CSI (median)",
    )

    plt.xlabel("Clear Sky Index (CSI)")
    plt.ylabel("Density")
    plt.title("CSI Distribution (CLEANED): Ground vs Satellite")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        logger.success(f"[PLOT] Saved CSI histogram to {out_path}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@app.command()
def main(
    clip_max: float = 2.0,
    resample_rule: str = "10min",
    night_ghi_clear_min: float = 20.0,
):
    logger.info("Post-processing interim data -> processed (clean)")

    # ---------------- Ground ----------------
    gproc = GroundPreprocessor(cfg_path=GROUND_DATASET_CONFIG)
    g_written = gproc.postprocess_interim_ground(
        in_dir=INTERIM_DATA_DIR / "ground",
        out_dir=PROCESSED_DATA_DIR / "ground",
        resample_rule=resample_rule,
        clip_max=clip_max,
        night_ghi_clear_min=night_ghi_clear_min,
    )
    logger.success(f"Ground cleaned files: {len(g_written)}")

    # ---------------- Satellite ----------------
    sproc = SatellitePreprocessor(
        cfg_path=SATELLITE_DATASET_CONFIG,
        raw_dir=RAW_DATA_DIR / "satellite",
        interim_dir=INTERIM_DATA_DIR / "satellite",
        processed_dir=PROCESSED_DATA_DIR / "satellite",
    )
    s_out = sproc.postprocess_interim_satellite(
        in_csv=INTERIM_DATA_DIR / "satellite" / "satellite_irradiance.csv",
        out_csv=PROCESSED_DATA_DIR / "satellite" / "satellite_irradiance_clean.csv",
        resample_rule=resample_rule,
        clip_max=clip_max,
        night_ghi_clear_min=night_ghi_clear_min,
    )
    logger.success(f"Satellite cleaned: {s_out}")

    # ---------------- Plot CSI ----------------
    plot_csi_histogram_cleaned(
        ground_dir=PROCESSED_DATA_DIR / "ground",
        satellite_csv=PROCESSED_DATA_DIR / "satellite" / "satellite_irradiance_clean.csv",
        station_code="PUY",
        bins=60,
        csi_range=(0.0, 1.5),
        out_path=PROCESSED_DATA_DIR / "plots" / "csi_hist_clean_ground_vs_sat.png",
    )


if __name__ == "__main__":
    app()



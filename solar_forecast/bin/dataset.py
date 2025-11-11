from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from loguru import logger
import typer
import yaml
import xarray as xr
import numpy as np

from solar_forecast.config.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, GROUND_DATASET_CONFIG, SATELLITE_DATASET_CONFIG

app = typer.Typer()

"""
Script to download MeteoSwiss ground station data and satellite irradiance data.
"""

# ---------------------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------------------

# Load YAML configuration
with open(GROUND_DATASET_CONFIG, "r") as f:
    GROUND_CFG = yaml.safe_load(f)

with open(SATELLITE_DATASET_CONFIG, "r") as f:
    SAT_CFG = yaml.safe_load(f)

COLUMNS = GROUND_CFG["columns"]
RAW_COLS = COLUMNS["raw"]
RENAMED_COLS = COLUMNS["renamed"]

TIMESTAMP_COL = RAW_COLS["timestamp"]
STATION_COL = RAW_COLS["station"]
IRRADIANCE_COL = RAW_COLS["irradiance"]
LAT_COL = RAW_COLS["lat"]
LON_COL = RAW_COLS["lon"]
ALT_COL = RAW_COLS["alt"]

RENAMED_TIMESTAMP_COL = RENAMED_COLS["timestamp"]
RENAMED_IRRADIANCE_COL = RENAMED_COLS["irradiance"]
RENAMED_LAT_COL = RENAMED_COLS["lat"]
RENAMED_LON_COL = RENAMED_COLS["lon"]
RENAMED_STATION_COL = RENAMED_COLS["station"]

SAT_VAR = SAT_CFG["structure"]["data_variable"]
LAT_RANGE = SAT_CFG["preprocessing"]["subset_region"]["lat_range"]
LON_RANGE = SAT_CFG["preprocessing"]["subset_region"]["lon_range"]


# Configuration
STATIONS_META_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv"
STATIONS_URL_TEMPLATE = (
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/"
    "{station_lower}/ogd-smn_{station_lower}_t_historical_2020-2029.csv"
)

RAW_GROUND_DIR = RAW_DATA_DIR / "ground"
RAW_SATELLITE_DIR = RAW_DATA_DIR / "satellite"
INTERIM_GROUND_DIR = INTERIM_DATA_DIR / "ground"
INTERIM_SATELLITE_DIR = INTERIM_DATA_DIR / "satellite"
for p in [RAW_GROUND_DIR, RAW_SATELLITE_DIR, INTERIM_GROUND_DIR, INTERIM_SATELLITE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

TEST_STATIONS = ["BIE", "PUY", "VEV", "ORO", "MLS", "MAH"]

# ---------------------------------------------------------------------
# GROUND DATA
# ---------------------------------------------------------------------

"""
Download and process ground station data from MeteoSwiss.

Steps:
1. Load metadata of all stations
2. For each station, download its historical CSV
3. Process the data (parse timestamps, clean, rename columns)
4. Save cleaned CSVs to INTERIM_DATA_DIR
5. Generate summary (with metadata info: lat, lon, etc.)

Configuration:
- Controlled via config/datasets/irradiance.yaml

Usage:
    python -m solar_forecast.dataset --test
"""

# Load metadata
def load_station_metadata() -> pd.DataFrame:
    logger.info("Loading MeteoSwiss station metadata...")
    stations_df = pd.read_csv(STATIONS_META_URL, sep=";", encoding="latin1")
    logger.info(f"{len(stations_df)} stations found.")
    return stations_df

# Get list of station codes
def get_station_codes(stations_df: pd.DataFrame) -> list[str]:
    if STATION_COL not in stations_df.columns:
        raise ValueError(f"Column '{STATION_COL}' not found in metadata.")
    return stations_df[STATION_COL].dropna().astype(str).tolist()


# Download 1 station file
def download_station_data(station_code: str) -> Path | None:
    st = station_code.lower()
    url = STATIONS_URL_TEMPLATE.format(station_lower=st)
    dest = RAW_GROUND_DIR / f"{st}_historical.csv"

    if dest.exists():
        logger.debug(f"{station_code}: already downloaded.")
        return dest

    resp = requests.get(url)
    if resp.status_code == 200:
        dest.write_bytes(resp.content)
        logger.success(f"Downloaded {station_code}")
        return dest
    elif resp.status_code == 404:
        logger.warning(f"No data found for {station_code}")
    else:
        logger.error(f"Error {resp.status_code} for {station_code}")
    return None


# Process 1 station file
def process_station_file(input_path: Path, output_dir: Path = INTERIM_GROUND_DIR) -> Path | None:
    try:
        df = pd.read_csv(input_path, sep=";")

        if TIMESTAMP_COL not in df.columns:
            raise ValueError(f"Missing {TIMESTAMP_COL} in {input_path.name}")

        df[RENAMED_TIMESTAMP_COL] = pd.to_datetime(
            df[TIMESTAMP_COL],
            format="%d.%m.%Y %H:%M",
            errors="coerce"
        )

        # Rename irradiance column if exists
        if IRRADIANCE_COL in df.columns:
            df = df.rename(columns={IRRADIANCE_COL: RENAMED_IRRADIANCE_COL})

        # Rename station column if exists
        if STATION_COL in df.columns:
            df = df.rename(columns={STATION_COL: RENAMED_STATION_COL})

        # Sort and clean
        df = df.sort_values(RENAMED_TIMESTAMP_COL).ffill().bfill()

        keep_cols = [
            c for c in [RENAMED_TIMESTAMP_COL, RENAMED_STATION_COL, RENAMED_IRRADIANCE_COL] if c in df.columns
        ]
        df = df[keep_cols]

        out_path = output_dir / input_path.name.replace("_historical", "_processed")
        df.to_csv(out_path, index=False)
        logger.success(f"Processed {input_path.name} → {out_path.name}")
        return out_path

    except Exception as e:
        logger.error(f"Failed to process {input_path.name}: {e}")
        return None


# Generate summary
def generate_summary(processed_files: list[Path], stations_meta: pd.DataFrame, output_dir: Path = INTERIM_GROUND_DIR):
    summary = []
    for f in processed_files:
        try:
            df = pd.read_csv(f)
            station = f.stem.split("_")[0].upper()
            meta_row = stations_meta.loc[stations_meta[STATION_COL] == station]
            summary.append({
                RENAMED_STATION_COL : station,
                "rows": len(df),
                "start": df[RENAMED_TIMESTAMP_COL].min(),
                "end": df[RENAMED_TIMESTAMP_COL].max(),
                RENAMED_LAT_COL: meta_row[LAT_COL].values[0] if not meta_row.empty else None,
                RENAMED_LON_COL: meta_row[LON_COL].values[0] if not meta_row.empty else None,
                "altitude": meta_row[ALT_COL].values[0] if not meta_row.empty else None,
            })
        except Exception as e:
            logger.warning(f"Could not summarize {f.name}: {e}")

    if summary:
        pd.DataFrame(summary).to_csv(output_dir / "ground_summary.csv", index=False)
        logger.success("Summary saved.")
    else:
        logger.warning("No valid processed files.")


# ---------------------------------------------------------------------
# SATELLITE DATA
# ---------------------------------------------------------------------

def load_satellite_data(raw_dir: Path) -> pd.DataFrame:
    """
    Load and combine all .nc satellite files in a directory using the YAML configuration.
    Returns a (time × pixels) irradiance matrix.
    """
    logger.info(f"Loading all satellite files from {raw_dir}...")

    files = sorted(raw_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {raw_dir}")

    logger.info(f"Found {len(files)} satellite files.")

    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal", coords="minimal", compat="override")

    if SAT_VAR not in ds:
        raise KeyError(f"No '{SAT_VAR}' variable found in the satellite dataset.")

    # Take only Switzerland region 
    if SAT_CFG["preprocessing"]["subset_region"]["enabled"]:
        lat_min, lat_max = SAT_CFG["preprocessing"]["subset_region"]["lat_range"]
        lon_min, lon_max = SAT_CFG["preprocessing"]["subset_region"]["lon_range"]
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    # Irradiance DataFrame
    irr = ds[SAT_VAR]
    df_irr = irr.to_dataframe().reset_index()

    # Get the 11x11 subset around the test station
    half_size = SAT_CFG["preprocessing"]["subset_station"]["size"] // 2
    lat_center, lon_center, _ = SAT_CFG["preprocessing"]["subset_station"]["coord"]

    # Find nearest grid point to the station
    unique_lats = np.sort(df_irr["lat"].unique())
    unique_lons = np.sort(df_irr["lon"].unique())
    lat_idx = (np.abs(np.sort(df_irr["lat"].unique()) - lat_center)).argmin()
    lon_idx = (np.abs(np.sort(df_irr["lon"].unique()) - lon_center)).argmin()
    
    # Define 11×11 window around it
    lat_start = max(lat_idx - half_size, 0)
    lat_end = lat_idx + half_size + 1
    lon_start = max(lon_idx - half_size, 0)
    lon_end = lon_idx + half_size + 1

    target_lats = unique_lats[lat_start:lat_end]
    target_lons = unique_lons[lon_start:lon_end]

    # Filter dataframe to keep only those pixels
    df_irr = df_irr[df_irr["lat"].isin(target_lats) & df_irr["lon"].isin(target_lons)]
    logger.info("11x11 pixel subset around PUY station extracted.")

    # Pivot into matrix (time × [lat, lon])
    time_col = SAT_CFG["output"]["columns"]["time_index"]
    spatial_cols = SAT_CFG["output"]["columns"]["spatial_index"]
    value_col = SAT_CFG["output"]["columns"]["value"]

    pivot = df_irr.pivot_table(index=time_col, columns=spatial_cols, values=value_col)
    logger.success(f"Satellite data loaded with shape {pivot.shape}")

    # save interim satellite data
    interim_sat_path = INTERIM_SATELLITE_DIR / "satellite_irradiance_matrix.csv"
    pivot.to_csv(interim_sat_path)
    logger.success(f"Processed satellite irradiance matrix saved to {interim_sat_path}")
    return pivot


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
@app.command()
def main(test: bool = typer.Option(False, help="Run in test mode (few stations only).")):

    # ground data pipeline
    logger.info("Starting MeteoSwiss dataset pipeline.")
    logger.info(f"Using config file: {GROUND_DATASET_CONFIG}")

    stations_meta = load_station_metadata()
    station_codes = get_station_codes(stations_meta)

    if test:
        station_codes = [s for s in TEST_STATIONS if s in station_codes]
        logger.info(f"Running in TEST mode ({len(station_codes)} stations).")

    processed_files = []
    for code in tqdm(station_codes, desc="Stations"):
        csv_path = download_station_data(code)
        if csv_path:
            processed = process_station_file(csv_path)
            if processed:
                processed_files.append(processed)

    generate_summary(processed_files, stations_meta)
    
    # satellite data pipeline 
    irr_matrix = load_satellite_data(Path(RAW_SATELLITE_DIR))
    logger.info(f"Satellite data matrix shape: {irr_matrix.shape}")

    logger.success("All data processed and saved.")


if __name__ == "__main__":
    app()

"""
Prepare model-ready tensors and adjacency matrix for spatio-temporal deep learning.

Steps:
1. Load cleaned (interim) ground station data
2. Merge all stations into a single time × station matrix
3. Split train/val/test (chronologically)
4. Fit StandardScaler on training set, transform all
5. Build adjacency matrix from coordinates
6. Window the time series into (X, Y)
7. Save all processed tensors to data/processed/

Usage:
    python -m solar_forecast.features
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
import typer
from scipy.spatial.distance import cdist
import yaml
import joblib
from sklearn.preprocessing import StandardScaler

from solar_forecast.config.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, GROUND_DATASET_CONFIG, SATELLITE_DATASET_CONFIG, PREPROCESSING_DATASET_CONFIG

app = typer.Typer()

# ---------------------------------------------------------------------
# CONFIGURATIONS
# ---------------------------------------------------------------------

META_PATH = INTERIM_DATA_DIR / "ground" / "ground_summary.csv"
PROCESSED_GROUND_DIR = PROCESSED_DATA_DIR / "ground"
PROCESSED_SAT_DIR = PROCESSED_DATA_DIR / "satellite"

# Load YAML configuration
with open(GROUND_DATASET_CONFIG, "r") as f:
    GROUND_CFG = yaml.safe_load(f)

with open(SATELLITE_DATASET_CONFIG, "r") as f:
    SAT_CFG = yaml.safe_load(f)

with open(PREPROCESSING_DATASET_CONFIG, "r") as f:
    PP_CFG = yaml.safe_load(f)

# ---------------------------------------------------------------------
# GROUND DATA
# ---------------------------------------------------------------------

# Load and merge station CSVs
def load_and_merge_data(interim_dir: Path, renamed_cols: dict, station_col: str) -> pd.DataFrame:
    logger.info(f"Loading processed ground data from {interim_dir}...")
    dfs = []
    for file in tqdm(list(interim_dir.glob("*_processed.csv")), desc="Stations"):
        df = pd.read_csv(file)
        if {renamed_cols["timestamp"], station_col, renamed_cols["irradiance"]}.issubset(df.columns):
            dfs.append(df)
        else:
            logger.warning(f"Missing columns in {file.name}, skipping.")
    if not dfs:
        raise RuntimeError("No valid processed CSVs found in interim directory.")

    all_df = pd.concat(dfs, ignore_index=True)
    pivot = all_df.pivot(
        index=renamed_cols["timestamp"],
        columns=station_col,
        values=renamed_cols["irradiance"]
    )

    pivot.index = pd.to_datetime(pivot.index, errors="coerce")
    pivot = pivot.sort_index().interpolate().ffill().bfill()
    logger.success(f"Merged {len(dfs)} stations → matrix shape {pivot.shape}")
    return pivot


# Train/val/test split (chronological)
def chronological_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# Scaling with StandardScaler
def scale_train_val_test(train_df, val_df, test_df, scaler_path: Path):
    scaler = StandardScaler()
    scaler.fit(train_df.values)

    train_scaled = pd.DataFrame(scaler.transform(train_df), index=train_df.index, columns=train_df.columns)
    val_scaled = pd.DataFrame(scaler.transform(val_df), index=val_df.index, columns=val_df.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df), index=test_df.index, columns=test_df.columns)

    joblib.dump(scaler, scaler_path)
    logger.success(f"StandardScaler fitted and saved → {scaler_path}")
    return train_scaled, val_scaled, test_scaled


# Build adjacency matrix
def build_adjacency(coords_df: pd.DataFrame, cfg_graph: dict) -> np.ndarray:
    logger.info("Building adjacency matrix from station coordinates...")
    sigma_km = cfg_graph.get("sigma_km", 50)
    max_dist_km = cfg_graph.get("max_dist_km", 150)
    lat_col = cfg_graph.get("lat_col", "lat")
    lon_col = cfg_graph.get("lon_col", "lon")

    coords = coords_df[[lat_col, lon_col]].to_numpy()
    dist = cdist(coords, coords, metric="euclidean") * 111  # degrees → km
    A = np.exp(-dist**2 / (2 * sigma_km**2))
    if max_dist_km is not None:
        A[dist > max_dist_km] = 0
    np.fill_diagonal(A, 0)
    logger.success(f"Adjacency matrix built: shape={A.shape}, mean weight={A.mean():.3f}")
    return A


# Create sequences efficiently
def create_sequences(X: np.ndarray, window: int, horizon: int, stride: int):
    """
    Vectorized windowing of time series.
    """
    n_time, n_nodes = X.shape
    n_samples = (n_time - window - horizon) // stride + 1

    X_seq = np.lib.stride_tricks.sliding_window_view(X, (window, n_nodes))[::stride, 0]
    Y_seq = np.lib.stride_tricks.sliding_window_view(X, (horizon, n_nodes))[window::stride, 0]

    X_seq = np.expand_dims(X_seq, -1)
    Y_seq = np.expand_dims(Y_seq, -1)
    return X_seq[:n_samples], Y_seq[:n_samples]


# Save all processed outputs
def save_processed(train, val, test, A, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    window_cfg = GROUND_DATASET_CONFIG["windowing"]

    window = window_cfg.get("window", 12)
    horizon = window_cfg.get("horizon", 1)
    stride = window_cfg.get("stride", 1)

    logger.info(f"Creating sequences (window={window}, horizon={horizon}, stride={stride})...")
    X_train, Y_train = create_sequences(train.values, window, horizon, stride)
    X_val, Y_val = create_sequences(val.values, window, horizon, stride)
    X_test, Y_test = create_sequences(test.values, window, horizon, stride)
    logger.success(f"Sequence shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "Y_train.npy", Y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "Y_val.npy", Y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "Y_test.npy", Y_test)
    np.save(output_dir / "A.npy", A)

    logger.success(f"Saved processed data to {output_dir}")


# ---------------------------------------------------------------------
# SATELLITE DATA
# ---------------------------------------------------------------------

# refaire des matrices 11x11 par date 



# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
@app.command()
def main():
    logger.info("Starting feature preparation pipeline...")
    renamed_cols = GROUND_CFG["columns"]["renamed"]
    station_col = renamed_cols["station"]
    graph_cfg = GROUND_CFG["graph"]
    splits_cfg = GROUND_CFG["splits"]
    
    # load ground and satellite data
    ground_dir = INTERIM_DATA_DIR / "ground"
    ground = load_and_merge_data(ground_dir, renamed_cols, station_col)
    satellite_dir = INTERIM_DATA_DIR / "satellite"
    satellite = pd.read_csv(satellite_dir/"satellite_irradiance_matrix.csv", skiprows=3, index_col=0)
    satellite.index = pd.to_datetime(satellite.index, errors="coerce")
    satellite = satellite.sort_index().interpolate().ffill().bfill()

    # take lowest sampling frequency between ground and satellite
    ground_freq = pd.to_timedelta(GROUND_CFG["dataset"]["frequency"])
    satellite_freq = pd.to_timedelta(SAT_CFG["dataset"]["frequency"])
    freq = max(ground_freq, satellite_freq)
    
    # resample both datasets to final_freq
    ground = ground.resample(freq).interpolate().ffill().bfill()
    satellite = satellite.resample(freq).interpolate().ffill().bfill()
    logger.info(f"Resampled both datasets to frequency: {freq}")    

    # keep only jan 2023
    date_start = PP_CFG["date_range"]["start"]
    date_end = PP_CFG["date_range"]["end"]
    ground = ground[(ground.index >= date_start) & (ground.index <= date_end)]
    satellite = satellite[(satellite.index >= date_start) & (satellite.index <= date_end)]
    logger.info(f"Filtered both datasets to date range: {date_start} to {date_end}")

    






    # Split and scale
    train_df, val_df, test_df = chronological_split(ground, splits_cfg["train_ratio"], splits_cfg["val_ratio"])
    scaler_path = PROCESSED_GROUND_DIR / "scaler.joblib"
    train_scaled, val_scaled, test_scaled = scale_train_val_test(train_df, val_df, test_df, scaler_path)

    # Build adjacency
    meta_df = pd.read_csv(META_PATH)
    meta_df = meta_df[meta_df[station_col].isin(ground.columns)]
    A = build_adjacency(meta_df, graph_cfg)

    # Save sequences
    save_processed(train_scaled, val_scaled, test_scaled, A, PROCESSED_GROUND_DIR)
    logger.success("Feature preparation complete. Data ready for training!")


if __name__ == "__main__":
    app()

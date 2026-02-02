from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib
import yaml


class SatellitePreprocessor:
    """
    Handles preprocessing of satellite-based irradiance data (NetCDF).
    Includes:
      - Loading .nc files
      - Extracting regions and patches
      - Saving to interim CSV matrix
      - Scaling, splitting, windowing
    """

    def __init__(self, cfg_path: Path, raw_dir: Path, interim_dir: Path, processed_dir: Path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)
        self.scaler = None

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    
    # Load .nc files 
    def load_from_nc(self) -> pd.DataFrame:
        logger.info(f"Loading satellite NetCDF files from {self.raw_dir}...")

        files = sorted(self.raw_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files found in {self.raw_dir}")
        logger.info(f"Found {len(files)} NetCDF files.")

        ds = xr.open_mfdataset(
            files, combine="by_coords", data_vars="minimal", coords="minimal", compat="override"
        )

        sat_var = self.cfg["structure"]["data_variable"]
        if sat_var not in ds:
            raise KeyError(f"No variable '{sat_var}' found in dataset.")

        # Switzerland region 
        subset_cfg = self.cfg["preprocessing"]["subset_region"]
        if subset_cfg.get("enabled", True):
            lat_min, lat_max = subset_cfg["lat_range"]
            lon_min, lon_max = subset_cfg["lon_range"]
            ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

        # Convert to DataFrame
        df_irr = ds[sat_var].to_dataframe().reset_index()

        # Extract patch around reference station
        station_cfg = self.cfg["preprocessing"]["subset_station"]
        half_size = station_cfg["size"] // 2
        lat_center, lon_center, _ = station_cfg["coord"]

        unique_lats = np.sort(df_irr["lat"].unique())
        unique_lons = np.sort(df_irr["lon"].unique())

        lat_idx = (np.abs(unique_lats - lat_center)).argmin()
        lon_idx = (np.abs(unique_lons - lon_center)).argmin()

        lat_start, lat_end = max(lat_idx - half_size, 0), lat_idx + half_size + 1
        lon_start, lon_end = max(lon_idx - half_size, 0), lon_idx + half_size + 1

        target_lats = unique_lats[lat_start:lat_end]
        target_lons = unique_lons[lon_start:lon_end]

        df_patch = df_irr[df_irr["lat"].isin(target_lats) & df_irr["lon"].isin(target_lons)]
        logger.info(f"Extracted {len(target_lats)}×{len(target_lons)} patch around reference station.")

        # Save to interim CSV
        path = self.interim_dir / "satellite_irradiance.csv"
        df_patch.to_csv(path, index=False)
        logger.success(f"Saved satellite CSV → {path}")

        return df_patch

    # Download CSV
    def redownload_csv(self, overwrite: bool = True) -> pd.DataFrame:
        csv_path = self.interim_dir / "satellite_irradiance.csv"
        logger.info(f"Loading existing CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df

    # Pivot to matrix
    def pivot_to_matrix(self, df_irr: pd.DataFrame) -> pd.DataFrame:
        out_cols = self.cfg["output"]["columns"]
        time_col, spatial_cols, value_col = (
            out_cols["time_index"],
            out_cols["spatial_index"],
            out_cols["value"],
        )

        pivot = df_irr.pivot_table(index=time_col, columns=spatial_cols, values=value_col)

        return pivot


    # # ------------------------------------------------------------------
    # # 3️⃣ TRAIN/VAL/TEST SPLIT + SCALING
    # # ------------------------------------------------------------------
    # def split(self, df):
    #     n = len(df)
    #     tr, vr = self.cfg["splits"]["train_ratio"], self.cfg["splits"]["val_ratio"]
    #     return df.iloc[:int(n*tr)], df.iloc[int(n*tr):int(n*(tr+vr))], df.iloc[int(n*(tr+vr)):]

    # def scale(self, train, val, test):
    #     self.scaler = StandardScaler().fit(train)
    #     transform = lambda d: pd.DataFrame(self.scaler.transform(d), index=d.index, columns=d.columns)
    #     joblib.dump(self.scaler, self.processed_dir / "scaler.joblib")
    #     logger.success("Scaler fitted and saved.")
    #     return transform(train), transform(val), transform(test)

    # # ------------------------------------------------------------------
    # # 4️⃣ WINDOWING (TEMPORAL SEQUENCES)
    # # ------------------------------------------------------------------
    # def create_sequences(self, X, window, horizon, stride):
    #     n_time, n_nodes = X.shape
    #     n_samples = (n_time - window - horizon) // stride + 1
    #     X_seq = np.lib.stride_tricks.sliding_window_view(X, (window, n_nodes))[::stride, 0]
    #     Y_seq = np.lib.stride_tricks.sliding_window_view(X, (horizon, n_nodes))[window::stride, 0]
    #     return np.expand_dims(X_seq, -1), np.expand_dims(Y_seq, -1)

    # def save(self, train, val, test):
    #     self.processed_dir.mkdir(parents=True, exist_ok=True)
    #     w, h, s = (self.cfg["windowing"][k] for k in ("window", "horizon", "stride"))
    #     Xt, Yt = self.create_sequences(train.values, w, h, s)
    #     Xv, Yv = self.create_sequences(val.values, w, h, s)
    #     Xte, Yte = self.create_sequences(test.values, w, h, s)
    #     np.save(self.processed_dir / "X_train.npy", Xt)
    #     np.save(self.processed_dir / "Y_train.npy", Yt)
    #     np.save(self.processed_dir / "X_val.npy", Xv)
    #     np.save(self.processed_dir / "Y_val.npy", Yv)
    #     np.save(self.processed_dir / "X_test.npy", Xte)
    #     np.save(self.processed_dir / "Y_test.npy", Yte)
    #     np.save(self.processed_dir / "A.npy", np.eye(Xt.shape[2]))  # placeholder adjacency
    #     logger.success(f"✅ Saved satellite tensors to {self.processed_dir}")

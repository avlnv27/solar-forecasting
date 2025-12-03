from pathlib import Path
import pandas as pd
import numpy as np
import requests
import yaml
import xarray as xr

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader

from typing import Optional

import torch
from torch import Tensor

from tsl.nn.utils import maybe_cat_exog
import torch
import math

maybe_cat_exog = maybe_cat_exog

def maybe_cat_emb(x: Tensor, emb: Optional[Tensor]):
    if emb is None:
        return x
    if emb.ndim < x.ndim:
        emb = emb[[None] * (x.ndim - emb.ndim)]
    emb = emb.expand(*x.shape[:-1], -1)
    return torch.cat([x, emb], dim=-1)


def maybe_cat_v(u: Tensor, v: Optional[Tensor]):
    return maybe_cat_emb(x=u, emb=v)


# -------------------------------------------------------------------
# GROUND PREPROCESSOR CLASS
# -------------------------------------------------------------------

class GroundPreprocessor:

    # ---------------------------------------------------------------
    # COLUMN HANDLER
    # ---------------------------------------------------------------
    class Columns:
        def __init__(self, cfg):
            raw = cfg["columns"]["raw"]
            renamed = cfg["columns"]["renamed"]

            # Raw
            self.timestamp = raw["timestamp"]
            self.station = raw["station"]
            self.irradiance = raw["irradiance"]
            self.lat = raw["lat"]
            self.lon = raw["lon"]
            self.alt = raw["alt"]

            # Renamed
            self.r_timestamp = renamed["timestamp"]
            self.r_station = renamed["station"]
            self.r_irradiance = renamed["irradiance"]
            self.r_lat = renamed["lat"]
            self.r_lon = renamed["lon"]
            self.r_alt = renamed["alt"]


    # ---------------------------------------------------------------
    # INIT
    # ---------------------------------------------------------------
    def __init__(self, cfg_path: Path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        # Directories
        self.raw_dir = Path(self.cfg["paths"]["raw_dir"])
        self.interim_dir = Path(self.cfg["paths"]["interim_dir"])
        self.processed_dir = Path(self.cfg["paths"]["processed_dir"])

        # Metadata tools
        self.meta_path = self.interim_dir / "ground_summary.csv"
        self.cols = self.Columns(self.cfg)

        # Data configuration
        self.frequency = pd.to_timedelta(self.cfg["dataset"]["frequency"])
        self.variables = self.cfg["dataset"]["variables"]

        # Windowing + graph + scaling
        self.window_cfg = self.cfg["windowing"]
        self.graph_cfg = self.cfg["graph"]
        self.scaling_cfg = self.cfg["scaling"]

        # Train/val/test
        self.train_ratio = self.cfg["splits"]["train_ratio"]
        self.val_ratio = self.cfg["splits"]["val_ratio"]
        self.test_ratio = self.cfg["splits"]["test_ratio"]

        # MeteoSwiss sources
        src = self.cfg.get("sources", {})
        self.meta_url = src.get("meta_url")
        self.station_url_tpl = src.get("station_url_template")
        self.test_stations = src.get("stations", [])

        # Scaling
        method = self.scaling_cfg.get("method", "standardize")
        if method == "standardize":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None


    # ---------------------------------------------------------------
    # LOAD STATION METADATA
    # ---------------------------------------------------------------
    def load_station_metadata(self) -> pd.DataFrame:
        logger.info("Loading MeteoSwiss station metadata...")
        df = pd.read_csv(self.meta_url, sep=";", encoding="latin1")
        logger.info(f"Loaded metadata with {len(df)} rows.")
        return df


    # ---------------------------------------------------------------
    # DOWNLOAD RAW CSVs
    # ---------------------------------------------------------------
    def download_all_stations(self) -> list[Path]:
        logger.info("Downloading ground station CSVs...")
        paths = []

        for code in self.test_stations:
            st = code.lower()
            url = self.station_url_tpl.format(station_lower=st)
            dest = self.raw_dir / f"{st}_historical.csv"

            if dest.exists():
                logger.debug(f"{code}: already exists.")
                paths.append(dest)
                continue

            r = requests.get(url)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                logger.success(f"Downloaded {code}")
                paths.append(dest)
            elif r.status_code == 404:
                logger.warning(f"No data for station {code}")
            else:
                logger.error(f"HTTP {r.status_code} for {code}")

        return paths


    # ---------------------------------------------------------------
    # PROCESS SINGLE STATION CSV
    # ---------------------------------------------------------------
    def process_station_file(self, input_path: Path) -> Path | None:
        try:
            c = self.cols
            df = pd.read_csv(input_path, sep=";")

            if c.timestamp not in df.columns:
                raise ValueError(f"Missing timestamp column in {input_path.name}")

            # Convert timestamps
            df[c.r_timestamp] = pd.to_datetime(
                df[c.timestamp], format="%d.%m.%Y %H:%M", errors="coerce"
            )

            # Rename irradiance + station columns
            if c.irradiance in df.columns:
                df = df.rename(columns={c.irradiance: c.r_irradiance})
            if c.station in df.columns:
                df = df.rename(columns={c.station: c.r_station})

            df = df.sort_values(c.r_timestamp).ffill().bfill()

            keep = [c.r_timestamp, c.r_station, c.r_irradiance]
            df = df[[col for col in keep if col in df.columns]]

            out = self.interim_dir / input_path.name.replace("_historical", "_processed")
            df.to_csv(out, index=False)
            logger.success(f"{input_path.name} processed → {out.name}")
            return out

        except Exception as e:
            logger.error(f"Error processing {input_path.name}: {e}")
            return None


    # ---------------------------------------------------------------
    # MERGE ALL PROCESSED STATION FILES
    # ---------------------------------------------------------------
    def load_and_merge(self) -> pd.DataFrame:
        c = self.cols
        logger.info("Merging processed ground data...")

        dfs = []
        for f in self.interim_dir.glob("*_processed.csv"):
            df = pd.read_csv(f)
            if {c.r_timestamp, c.r_station, c.r_irradiance}.issubset(df.columns):
                dfs.append(df)

        merged = pd.concat(dfs)
        pivot = merged.pivot(index=c.r_timestamp, columns=c.r_station,
                             values=c.r_irradiance)

        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index().interpolate().ffill().bfill()

        logger.success(f"Ground merged: {pivot.shape}")
        return pivot


    # ---------------------------------------------------------------
    # STATIC GRAPH UTILITIES
    # ---------------------------------------------------------------

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 +
             math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
        return 2 * R * math.asin(math.sqrt(a))

    @staticmethod
    def build_distance_matrix(coords):
        n = len(coords)
        D = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                D[i, j] = GroundPreprocessor.haversine(*coords[i], *coords[j])
        return D

    @staticmethod
    def build_knn_graph(D, k):
        n = D.shape[0]
        edges = []
        for i in range(n):
            nearest = torch.argsort(D[i])[:k+1]
            for j in nearest:
                if i != j:
                    edges.append((i, j))
        return torch.tensor(edges).t().long()

    @staticmethod
    def compute_edge_weight(D, edge_index, sigma=20):
        i, j = edge_index
        return torch.exp(-D[i, j] / sigma)


    # ---------------------------------------------------------------
    # MAIN GRAPH BUILDER
    # ---------------------------------------------------------------
        # ---------------------------------------------------------------
    # MAIN GRAPH BUILDER (YAML-COMPATIBLE VERSION)
    # ---------------------------------------------------------------
    def build_graph(self):
        cfg = self.graph_cfg

        # Load metadata to get lat/lon
        meta = self.load_station_metadata()

        # Use YAML station list
        station_col = self.cols.station  # ex: "station_abbr"
        selected = meta[meta[station_col].isin(self.test_stations)]

        # Extract coordinates (raw YAML fields)
        coords = list(zip(
            selected[self.cols.lat],
            selected[self.cols.lon],
        ))

        # Distance matrix in km
        D = self.build_distance_matrix(coords)

        # Graph construction rule based on YAML:
        # Keep edges where D[i,j] <= max_dist_km
        max_dist = cfg.get("max_dist_km", 150)
        n = len(coords)

        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and D[i, j] <= max_dist:
                    edges.append((i, j))

        if len(edges) == 0:
            raise ValueError(
                f"No edges found: max_dist_km={max_dist} is too small "
                f"for {n} stations."
            )

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Edge weights : Gaussian decay with sigma_km
        sigma = cfg.get("sigma_km", 50)
        i, j = edge_index
        edge_weight = torch.exp(-D[i, j] / sigma)

        logger.success(
            f"Graph built: {edge_index.shape[1]} edges "
            f"(max_dist_km={max_dist}, sigma_km={sigma})."
        )

        return edge_index, edge_weight, coords














# SATELLITE PREPROCESSOR
class SatellitePreprocessor:
    """
    Converts NetCDF → (T, H, W) tensor
    """

    def __init__(self, cfg_path: Path, raw_dir: Path, interim_dir: Path, processed_dir: Path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


    # Load all .nc and extract patch
    def load_from_nc(self) -> pd.DataFrame:
        logger.info(f"Loading satellite NetCDF files from {self.raw_dir}...")

        files = sorted(self.raw_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError("No .nc files found.")

        ds = xr.open_mfdataset(files, combine="by_coords")

        sat_var = self.cfg["structure"]["data_variable"]
        if sat_var not in ds:
            raise KeyError(f"Variable {sat_var} not found.")

        # Subset region
        sub = self.cfg["preprocessing"]["subset_region"]
        ds = ds.sel(
            lat=slice(sub["lat_range"][0], sub["lat_range"][1]),
            lon=slice(sub["lon_range"][0], sub["lon_range"][1])
        )

        # Convert to dataframe
        df = ds[sat_var].to_dataframe().reset_index()

        # Extract patch around reference station
        patch_cfg = self.cfg["preprocessing"]["subset_station"]
        lat_center, lon_center, _ = patch_cfg["coord"]
        half_size = patch_cfg["size"] // 2

        unique_lats = np.sort(df["lat"].unique())
        unique_lons = np.sort(df["lon"].unique())

        lat_idx = (np.abs(unique_lats - lat_center)).argmin()
        lon_idx = (np.abs(unique_lons - lon_center)).argmin()

        lat_start, lat_end = lat_idx - half_size, lat_idx + half_size + 1
        lon_start, lon_end = lon_idx - half_size, lon_idx + half_size + 1

        target_lats = unique_lats[lat_start:lat_end]
        target_lons = unique_lons[lon_start:lon_end]

        df_patch = df[df["lat"].isin(target_lats) &
                      df["lon"].isin(target_lons)]

        logger.info(f"Patch size: {len(target_lats)}×{len(target_lons)}")

        # Save
        out = self.interim_dir / "satellite_irradiance.csv"
        df_patch.to_csv(out, index=False)
        logger.success(f"Saved → {out}")
        return df_patch
    

    # Pivot to (T, H, W)
    def pivot_to_tensor(self, df_patch: pd.DataFrame) -> torch.Tensor:

        cfg = self.cfg["output"]["columns"]
        time_col, spatial_col, value_col = (
            cfg["time_index"],
            cfg["spatial_index"],
            cfg["value"]
        )

        pivot = df_patch.pivot_table(
            index=time_col,
            columns=spatial_col,
            values=value_col
        )

        T = pivot.shape[0]
        HW = pivot.shape[1]
        side = int(np.sqrt(HW))

        if side * side != HW:
            raise ValueError("Satellite patch is not square.")

        arr = pivot.values.reshape(T, side, side)

        return torch.tensor(arr, dtype=torch.float32)


# DATASET + DATALOADER
class ForecastDataset(Dataset):
    def __init__(self, sat_data, ground_data, targets, past_steps, future_steps):
        self.sat = sat_data            # (T, H, W)
        self.ground = ground_data      # (T, N, 1)
        self.targets = targets         # (T,)
        self.T = len(sat_data)
        self.past = past_steps
        self.future = future_steps

    def __len__(self):
        return self.T - self.past - self.future

    def __getitem__(self, idx):

        # --------------------- Satellite ---------------------
        # Select past_frames (past_steps, H, W)
        sat_win = self.sat[idx : idx + self.past]           # (past, H, W)
        sat_win = torch.tensor(sat_win, dtype=torch.float32)

        # Final shape = (C=past_steps, H, W)
        X_sat = sat_win                                     # naming clean

        # --------------------- Ground ------------------------
        X_ground = torch.tensor(self.ground[idx:idx+self.past], dtype=torch.float32).unsqueeze(-1)

        # --------------------- Target ------------------------
        y = torch.tensor(
            self.targets[idx + self.past : idx + self.past + self.future],
            dtype=torch.float32
        )

        return {
            "satellite": X_sat,        # (past_steps, H, W)
            "ground": X_ground,        # (past_steps, N, 1)
            "target": y                # (future_steps,)
        }


def make_dataloader(
    sat_data,
    ground_data,
    target_data,
    cfg,
    batch_size=32,
    shuffle=True
):
    past_steps = cfg["past_timesteps"]
    future_steps = cfg["future_timesteps"]

    dataset = ForecastDataset(
        sat_data=sat_data,
        ground_data=ground_data,
        targets=target_data,
        past_steps=past_steps,
        future_steps=future_steps,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )

    return loader

# solar_forecast/nn/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import math
import yaml
import requests
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from tsl.nn.utils import maybe_cat_exog as _maybe_cat_exog

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
maybe_cat_exog = _maybe_cat_exog


def maybe_cat_emb(x: Tensor, emb: Optional[Tensor]):
    if emb is None:
        return x
    if emb.ndim < x.ndim:
        emb = emb[[None] * (x.ndim - emb.ndim)]
    emb = emb.expand(*x.shape[:-1], -1)
    return torch.cat([x, emb], dim=-1)


def maybe_cat_v(u: Tensor, v: Optional[Tensor]):
    return maybe_cat_emb(x=u, emb=v)


def _drop_csi_clipped(
    df: pd.DataFrame,
    csi_cols: list[str],
    clip_max: float,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Drops rows where ANY CSI column is at the clip maximum (≈ clip_max).
    We treat values >= clip_max - eps as clipped.
    """
    if not csi_cols:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for col in csi_cols:
        if col in df.columns:
            mask |= df[col].astype(float).to_numpy() >= (float(clip_max) - eps)
    return df.loc[~mask]


def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Ensures a tz-aware UTC DatetimeIndex.
    - If idx is tz-naive: interpret as UTC.
    - If idx has tz: convert to UTC.
    """
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _parse_time_col_to_utc_naive(s: pd.Series) -> pd.Series:
    """
    Parse a datetime column as UTC and return tz-naive UTC (for CSV + merges).
    Robust if s is already tz-naive or tz-aware.
    """
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)


def _cfg_get(cfg: dict, path: Iterable[str], default=None):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ---------------------------------------------------------------------
# Gets Clear-sky GHI to process irradiances as CSI (pvlib)
# ---------------------------------------------------------------------
class ClearSkyIndexLoader:
    """
    Clear-sky + CSI using pvlib (no API calls).

    Requirements:
      pip install pvlib
    """

    def __init__(self, tz: str = "UTC"):
        self.tz = tz

    def clearsky_ghi(
        self,
        lat: float,
        lon: float,
        times: pd.DatetimeIndex,
        altitude: float | None = None,
        model: str = "ineichen",
    ) -> pd.Series:
        """
        Returns clear-sky GHI [W/m^2] for the given times (tz-aware UTC index).
        """
        try:
            from pvlib.location import Location
        except ImportError as e:
            raise ImportError("pvlib is not installed. Run: pip install pvlib") from e

        times_utc = _ensure_utc_index(times)
        loc = Location(latitude=float(lat), longitude=float(lon), tz="UTC", altitude=altitude)
        cs = loc.get_clearsky(times_utc, model=model)  # ghi, dni, dhi
        return cs["ghi"].rename("ghi_clear")

    def compute_csi(
        self,
        lat: float,
        lon: float,
        observed: pd.Series,  # indexed by datetime (UTC-naive or tz-aware)
        altitude: float | None = None,
        model: str = "ineichen",
        ghi_clear_min: float = 50.0,
        clip_max: float = 2.0,
        fillna: float = 0.0,
    ) -> pd.DataFrame:
        """
        Returns DataFrame (index tz-aware UTC):
          ghi_clear, ghi_obs, clear_sky_index
        """
        obs = observed.copy()
        obs.index = pd.to_datetime(obs.index)

        obs_idx_utc = _ensure_utc_index(obs.index)
        obs.index = obs_idx_utc

        ghi_clear = self.clearsky_ghi(
            lat=lat, lon=lon, times=obs_idx_utc, altitude=altitude, model=model
        ).reindex(obs_idx_utc)

        df = pd.DataFrame({"ghi_clear": ghi_clear, "ghi_obs": obs.astype(float)})

        denom = df["ghi_clear"].where(df["ghi_clear"] >= float(ghi_clear_min), np.nan)
        csi = (df["ghi_obs"] / denom).replace([np.inf, -np.inf], np.nan)

        if fillna is not None:
            csi = csi.fillna(float(fillna))
        csi = csi.clip(lower=0.0, upper=float(clip_max))

        df["clear_sky_index"] = csi
        return df


# ---------------------------------------------------------------------
# Ground preprocessing 
# ---------------------------------------------------------------------
class GroundPreprocessor:
    class Columns:
        def __init__(self, cfg):
            raw = cfg["columns"]["raw"]
            renamed = cfg["columns"]["renamed"]

            self.timestamp = raw["timestamp"]
            self.station = raw["station"]
            self.irradiance = raw["irradiance"]
            self.lat = raw["lat"]
            self.lon = raw["lon"]
            self.alt = raw["alt"]

            # Convention: processed CSV uses renamed keys
            self.r_time = renamed["timestamp"]
            self.r_station = renamed["station"]
            self.r_irradiance = renamed["irradiance"]
            self.r_lat = renamed.get("lat", "lat")
            self.r_lon = renamed.get("lon", "lon")
            self.r_alt = renamed.get("alt", "alt")

    def __init__(self, cfg_path: Path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = Path(self.cfg["paths"]["raw_dir"])
        self.interim_dir = Path(self.cfg["paths"]["interim_dir"])
        self.processed_dir = Path(self.cfg["paths"]["processed_dir"])

        self.meta_path = self.interim_dir / "ground_summary.csv"
        self.cols = self.Columns(self.cfg)

        self.frequency = pd.to_timedelta(self.cfg["dataset"]["frequency"])
        self.timezone = self.cfg["dataset"].get("timezone", "UTC")

        self.graph_cfg = self.cfg.get("graph", {})

        src = self.cfg.get("sources", {})
        self.meta_url = src.get("meta_url")
        self.station_url_tpl = src.get("station_url_template")
        self.test_stations = src.get("stations", [])

        self.csi_cfg = self.cfg.get("csi", {})
        # observed, ghi_clear, csi, ghi_clear_min, clip_max, fillna, model
        self.csi_obs_col = self.csi_cfg.get("observed", self.cols.r_irradiance)
        self.csi_clear_col = self.csi_cfg.get("ghi_clear", "ghi_clear")
        self.csi_col = self.csi_cfg.get("csi", f"{self.csi_obs_col}_csi")

    def load_station_metadata(self) -> pd.DataFrame:
        logger.info("Loading MeteoSwiss station metadata...")
        df = pd.read_csv(self.meta_url, sep=";", encoding="latin1")
        logger.info(f"Loaded metadata with {len(df)} rows.")
        return df

    def download_all_stations(self) -> list[Path]:
        logger.info("Downloading ground station CSVs...")
        paths: list[Path] = []
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

    def process_station_file(self, input_path: Path) -> Path | None:
        """
        MeteoSwiss reference timestamps are UTC.
        We parse them as UTC and store as tz-naive UTC in the processed CSV.
        Column names are taken from YAML (renamed.*).
        """
        try:
            c = self.cols
            df = pd.read_csv(input_path, sep=";")

            if c.timestamp not in df.columns:
                raise ValueError(f"Missing timestamp column in {input_path.name}")

            df[c.r_time] = pd.to_datetime(
                df[c.timestamp],
                format="%d.%m.%Y %H:%M",
                errors="coerce",
                utc=True,
            ).dt.tz_convert("UTC").dt.tz_localize(None)

            if c.irradiance in df.columns:
                df = df.rename(columns={c.irradiance: c.r_irradiance})
            if c.station in df.columns:
                df = df.rename(columns={c.station: c.r_station})

            df = df.sort_values(c.r_time).ffill().bfill()

            keep = [c.r_time, c.r_station, c.r_irradiance]
            df = df[[col for col in keep if col in df.columns]]

            out = self.interim_dir / input_path.name.replace("_historical", "_processed")
            df.to_csv(out, index=False)
            logger.success(f"{input_path.name} processed → {out.name}")
            return out

        except Exception as e:
            logger.error(f"Error processing {input_path.name}: {e}")
            return None

    def generate_summary(self, processed_files: list[Path], stations_meta: pd.DataFrame):
        c = self.cols
        summary = []
        for f in processed_files:
            try:
                df = pd.read_csv(f)
                station = f.stem.split("_")[0].upper()
                meta_row = stations_meta.loc[stations_meta[c.station] == station]
                summary.append(
                    {
                        c.r_station: station,
                        "rows": len(df),
                        "start": df[c.r_time].min(),
                        "end": df[c.r_time].max(),
                        c.r_lat: meta_row[c.lat].values[0] if not meta_row.empty else None,
                        c.r_lon: meta_row[c.lon].values[0] if not meta_row.empty else None,
                        c.r_alt: meta_row[c.alt].values[0] if not meta_row.empty else None,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not summarize {f.name}: {e}")

        if summary:
            pd.DataFrame(summary).to_csv(self.meta_path, index=False)
            logger.success("Ground summary saved.")
        else:
            logger.warning("No valid processed files found.")

    def add_csi_to_processed_files(
        self,
        processed_files: list[Path],
        stations_meta: pd.DataFrame,
    ) -> None:
        """
        Adds pvlib clear-sky GHI + CSI to each station *_processed.csv.

        Column names and defaults are read from YAML:
          csi:
            observed: <col>
            ghi_clear: <col>
            csi: <col>
            ghi_clear_min: ...
            clip_max: ...
            fillna: ...
            model: ...
        """
        c = self.cols
        meta = stations_meta.set_index(c.station)
        cs = ClearSkyIndexLoader(tz=self.timezone)

        # YAML defaults (allow override by args)
        clip_max = self.csi_cfg.get("clip_max", 2.0)
        fillna = self.csi_cfg.get("fillna", 0.0)
        ghi_clear_min = self.csi_cfg.get("ghi_clear_min", 20.0)
        model = self.csi_cfg.get("model", "ineichen")

        obs_col = self.csi_obs_col
        ghi_clear_col = self.csi_clear_col
        csi_col = self.csi_col

        for f in processed_files:
            if f is None:
                continue
            f = Path(f)
            if not f.exists():
                continue

            df = pd.read_csv(f)
            required = {c.r_time, c.r_station, c.r_irradiance}
            if not required.issubset(df.columns):
                logger.warning(f"[CSI-GROUND] Missing required cols in {f.name}, skip.")
                continue

            df[c.r_time] = pd.to_datetime(df[c.r_time])
            df = df.sort_values(c.r_time)

            st = str(df[c.r_station].iloc[0]).upper()
            if st not in meta.index:
                logger.warning(f"[CSI-GROUND] No metadata for station {st}, skip.")
                continue

            lat = pd.to_numeric(meta.loc[st, c.lat], errors="coerce")
            lon = pd.to_numeric(meta.loc[st, c.lon], errors="coerce")
            alt = pd.to_numeric(meta.loc[st, c.alt], errors="coerce") if c.alt in meta.columns else np.nan

            if not np.isfinite(lat) or not np.isfinite(lon):
                logger.error(f"[CSI-GROUND] Invalid lat/lon for {st}: lat={lat} lon={lon} -> skip.")
                continue

            altitude = float(alt) if np.isfinite(alt) else None

            # Observed series: always from df column obs_col (from YAML)
            if obs_col not in df.columns:
                logger.warning(f"[CSI-GROUND] obs_col='{obs_col}' not in {f.name}, skip.")
                continue

            obs = df.set_index(c.r_time)[obs_col].astype(float)

            try:
                out = cs.compute_csi(
                    lat=float(lat),
                    lon=float(lon),
                    altitude=altitude,
                    observed=obs,
                    model=model,
                    ghi_clear_min=ghi_clear_min,
                    clip_max=clip_max,
                    fillna=fillna,
                )

                idx = out.index.tz_localize(None)  # store as tz-naive UTC
                tmp = pd.DataFrame(
                    {
                        c.r_time: idx,
                        ghi_clear_col: out["ghi_clear"].to_numpy(),
                        csi_col: out["clear_sky_index"].to_numpy(),
                    }
                )

                df = df.merge(tmp, on=c.r_time, how="left")
                df.to_csv(f, index=False)
                logger.success(f"[CSI-GROUND] Added {ghi_clear_col} + {csi_col} to {f.name}")

            except Exception as e:
                logger.error(f"[CSI-GROUND] Failed for {st} ({f.name}): {e}")

    def postprocess_interim_ground(
        self,
        in_dir: Path | None = None,
        out_dir: Path | None = None,
        resample_rule: str = "10min",
        clip_max: Optional[float] = None,
        night_ghi_clear_min: Optional[float] = None,
    ) -> list[Path]:
        """
        Reads each *_processed.csv from interim ground dir, then:
        1) drops rows where CSI is clipped at clip_max
        2) resamples to resample_rule (mean)
        3) drops night using ghi_clear >= night_ghi_clear_min
        Writes cleaned files to out_dir with suffix *_clean.csv.

        CSI column + ghi_clear column come from YAML.
        """
        c = self.cols
        in_dir = Path(in_dir) if in_dir is not None else self.interim_dir
        out_dir = Path(out_dir) if out_dir is not None else self.processed_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        clip_max = float(clip_max if clip_max is not None else self.csi_cfg.get("clip_max", 2.0))
        ghi_clear_col = self.csi_clear_col
        csi_col = self.csi_col

        # night filter threshold (keep rows with clearsky >= threshold)
        night_ghi_clear_min = float(
            night_ghi_clear_min if night_ghi_clear_min is not None else self.csi_cfg.get("ghi_clear_min", 20.0)
        )

        written: list[Path] = []
        for f in sorted(in_dir.glob("*_processed.csv")):
            try:
                df = pd.read_csv(f)
                required = {c.r_time, c.r_station, self.csi_obs_col}
                if not required.issubset(df.columns):
                    logger.warning(f"[POST-GROUND] Missing required columns in {f.name}, skip.")
                    continue

                df[c.r_time] = pd.to_datetime(df[c.r_time])
                df = df.sort_values(c.r_time)

                if csi_col in df.columns:
                    df = _drop_csi_clipped(df, [csi_col], clip_max=clip_max)
                else:
                    logger.warning(f"[POST-GROUND] No CSI column '{csi_col}' in {f.name} -> clip filter skipped.")

                df = df.set_index(c.r_time)

                station = (
                    str(df[c.r_station].iloc[0])
                    if c.r_station in df.columns
                    else f.stem.split("_")[0].upper()
                )

                # resample numeric columns
                df_num = df.apply(pd.to_numeric, errors="ignore")
                df_rs = df_num.resample(resample_rule).mean(numeric_only=True)

                # night filter
                if ghi_clear_col in df_rs.columns:
                    df_rs = df_rs[df_rs[ghi_clear_col] >= float(night_ghi_clear_min)]
                else:
                    logger.warning(f"[POST-GROUND] No clear-sky column '{ghi_clear_col}' in {f.name} -> night filter skipped.")

                df_rs[c.r_station] = station
                df_rs = df_rs.reset_index()

                out_path = out_dir / f"{f.stem}_clean.csv"
                df_rs.to_csv(out_path, index=False)
                written.append(out_path)

                logger.success(
                    f"[POST-GROUND] {f.name}: -> {out_path.name} "
                    f"(rows {len(df)} -> {len(df_rs)})"
                )
            except Exception as e:
                logger.error(f"[POST-GROUND] Failed on {f.name}: {e}")

        return written

    # ---------------- Graph utilities ----------------
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))

    @staticmethod
    def build_distance_matrix(coords):
        n = len(coords)
        D = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                D[i, j] = GroundPreprocessor.haversine(*coords[i], *coords[j])
        return D

    def build_graph(self):
        cfg = self.graph_cfg
        meta = self.load_station_metadata()
        station_col = self.cols.station
        selected = meta[meta[station_col].isin(self.test_stations)]

        coords = list(zip(selected[self.cols.lat], selected[self.cols.lon]))
        D = self.build_distance_matrix(coords)

        max_dist = float(cfg.get("max_dist_km", 150))
        n = len(coords)

        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and D[i, j] <= max_dist:
                    edges.append((i, j))

        if len(edges) == 0:
            raise ValueError(f"No edges found: max_dist_km={max_dist} too small for {n} stations.")

        edge_index = torch.tensor(edges, dtype=torch.long).t()

        sigma = float(cfg.get("sigma_km", 50))
        i, j = edge_index
        edge_weight = torch.exp(-D[i, j] / sigma)

        logger.success(
            f"Graph built: {edge_index.shape[1]} edges (max_dist_km={max_dist}, sigma_km={sigma})."
        )
        return edge_index, edge_weight, coords


# ---------------------------------------------------------------------
# Satellite preprocessing 
# ---------------------------------------------------------------------
class SatellitePreprocessor:
    def __init__(self, cfg_path: Path, raw_dir: Path, interim_dir: Path, processed_dir: Path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)

        self.timezone = self.cfg["dataset"].get("timezone", "UTC")

        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Column mapping raw -> renamed (like ground)
        self.raw_cols = self.cfg["columns"]["raw"]
        self.ren_cols = self.cfg["columns"]["renamed"]

        self.col_time = self.ren_cols["timestamp"]
        self.col_lat = self.ren_cols["lat"]
        self.col_lon = self.ren_cols["lon"]
        self.col_irr = self.ren_cols["irradiance"]

        # value in NetCDF/df before renaming
        self.col_value_raw = self.raw_cols["value"]

        # CSI config (no hardcoded names)
        self.csi_cfg = self.cfg.get("csi", {})
        self.csi_obs_col = self.csi_cfg.get("observed", self.col_irr)
        self.csi_clear_col = self.csi_cfg.get("ghi_clear", "ghi_clear_center")
        self.csi_col = self.csi_cfg.get("csi", f"{self.csi_obs_col}_csi")

        # Patch cfg
        self.patch_cfg = self.cfg.get("preprocessing", {}).get("subset_station", {})
        self.region_cfg = self.cfg.get("preprocessing", {}).get("subset_region", {})

    def load_from_nc(self) -> pd.DataFrame:
        """
        Loads NetCDFs -> subset region -> extract patch -> write interim CSV.
        Output columns are standardized (renamed convention):
          timestamp, lat, lon, irradiance
        """
        logger.info(f"Loading satellite NetCDF files from {self.raw_dir}...")

        files = sorted(self.raw_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError("No .nc files found.")

        ds = xr.open_mfdataset(files, combine="by_coords", data_vars="all")

        # raw var name from YAML
        value_var = self.raw_cols["value"]
        if value_var not in ds.data_vars and value_var not in ds:
            # If user put "value" as SIS but actual var differs, show helpful error
            raise KeyError(f"Satellite raw.value='{value_var}' not found in NetCDF vars={list(ds.data_vars)}")

        # subset region
        if self.region_cfg:
            lat_range = self.region_cfg.get("lat_range")
            lon_range = self.region_cfg.get("lon_range")
            if lat_range and lon_range:
                ds = ds.sel(
                    lat=slice(lat_range[0], lat_range[1]),
                    lon=slice(lon_range[0], lon_range[1]),
                )

        df = ds[value_var].to_dataframe().reset_index()

        # rename raw -> renamed
        # raw expected: time/lat/lon/value (value_var)
        rename_map = {
            self.raw_cols["timestamp"]: self.col_time,
            self.raw_cols["lat"]: self.col_lat,
            self.raw_cols["lon"]: self.col_lon,
            self.raw_cols["value"]: self.col_irr, 
        }
        df = df.rename(columns=rename_map)

        # enforce UTC-naive timestamps
        df[self.col_time] = _parse_time_col_to_utc_naive(df[self.col_time])

        # Extract patch around station
        lat_center, lon_center, _ = self.patch_cfg.get("coord", [None, None, None])
        size = int(self.patch_cfg.get("size", 11))
        if lat_center is None or lon_center is None:
            raise KeyError("Missing preprocessing.subset_station.coord in satellite YAML")

        half_size = size // 2

        unique_lats = np.sort(df[self.col_lat].unique())
        unique_lons = np.sort(df[self.col_lon].unique())

        lat_idx = (np.abs(unique_lats - float(lat_center))).argmin()
        lon_idx = (np.abs(unique_lons - float(lon_center))).argmin()

        lat_start, lat_end = lat_idx - half_size, lat_idx + half_size + 1
        lon_start, lon_end = lon_idx - half_size, lon_idx + half_size + 1

        lat_start = max(lat_start, 0)
        lon_start = max(lon_start, 0)
        lat_end = min(lat_end, len(unique_lats))
        lon_end = min(lon_end, len(unique_lons))

        target_lats = unique_lats[lat_start:lat_end]
        target_lons = unique_lons[lon_start:lon_end]

        df_patch = df[df[self.col_lat].isin(target_lats) & df[self.col_lon].isin(target_lons)]

        logger.info(f"Patch size: {len(target_lats)}×{len(target_lons)}")

        out = self.interim_dir / "satellite_irradiance.csv"
        df_patch.to_csv(out, index=False)
        logger.success(f"Saved → {out}")
        return df_patch

    def pivot_to_tensor(self, df_patch: pd.DataFrame) -> torch.Tensor:
        """
        Build (T,H,W) tensor using renamed convention:
          time index = renamed.time
          columns = [renamed.lat, renamed.lon]
          values = renamed.irradiance
        """
        pivot = df_patch.pivot_table(
            index=self.col_time,
            columns=[self.col_lat, self.col_lon],
            values=self.col_irr,
        )

        T = pivot.shape[0]
        HW = pivot.shape[1]
        side = int(np.sqrt(HW))
        if side * side != HW:
            raise ValueError(f"Satellite patch is not square: HW={HW} -> side≈{side}")

        arr = pivot.values.reshape(T, side, side)
        return torch.tensor(arr, dtype=torch.float32)

    def add_csi_to_interim_csv(self) -> None:
        """
        Adds clear-sky + CSI to interim CSV.
        All column names + thresholds come from YAML `csi`.
        """
        csv_path = self.interim_dir / "satellite_irradiance.csv"
        if not csv_path.exists():
            logger.warning("[CSI-SAT] satellite_irradiance.csv not found, skip.")
            return

        sdf = pd.read_csv(csv_path)

        obs_col = self.csi_obs_col
        clear_col = self.csi_clear_col
        csi_col = self.csi_col

        clip_max = float(self.csi_cfg.get("clip_max", 2.0))
        fillna = float(self.csi_cfg.get("fillna", 0.0))
        ghi_clear_min = float(self.csi_cfg.get("ghi_clear_min", 20.0))
        model = str(self.csi_cfg.get("model", "ineichen"))

        if self.col_time not in sdf.columns or obs_col not in sdf.columns:
            logger.warning(f"[CSI-SAT] Missing columns '{self.col_time}'/'{obs_col}' in {csv_path.name}, skip.")
            return

        sdf[self.col_time] = _parse_time_col_to_utc_naive(sdf[self.col_time])
        sdf = sdf.sort_values(self.col_time)

        lat_center, lon_center, _ = self.patch_cfg.get("coord", [None, None, None])
        if lat_center is None or lon_center is None:
            raise KeyError("Missing preprocessing.subset_station.coord in satellite YAML")

        unique_times = pd.DatetimeIndex(pd.unique(sdf[self.col_time])).sort_values()
        cs = ClearSkyIndexLoader(tz=self.timezone)

        try:
            ghi_clear = cs.clearsky_ghi(
                lat=float(lat_center),
                lon=float(lon_center),
                times=unique_times,
                altitude=None,
                model=model,
            )

            ghi_clear_map = ghi_clear.copy()
            if getattr(ghi_clear_map.index, "tz", None) is not None:
                ghi_clear_map.index = ghi_clear_map.index.tz_localize(None)

            sdf[clear_col] = sdf[self.col_time].map(ghi_clear_map)

            denom = sdf[clear_col].where(sdf[clear_col] >= float(ghi_clear_min), np.nan)

            csi = (sdf[obs_col].astype(float) / denom).replace([np.inf, -np.inf], np.nan)
            csi = csi.fillna(float(fillna)).clip(lower=0.0, upper=float(clip_max))

            sdf[csi_col] = csi
            sdf.to_csv(csv_path, index=False)
            logger.success(f"[CSI-SAT] Added {clear_col} + {csi_col} to {csv_path.name}")

        except Exception as e:
            logger.error(f"[CSI-SAT] Failed: {e}")

    def postprocess_interim_satellite(
        self,
        in_csv: Path | None = None,
        out_csv: Path | None = None,
        resample_rule: str = "10min",
        clip_max: Optional[float] = None,
        night_ghi_clear_min: Optional[float] = None,
    ) -> Path | None:
        """
        Clean interim satellite CSV:
        1) drop clipped CSI
        2) resample per pixel (and FILL gaps to keep a dense grid)
        3) drop night via clear-sky column
        Column names come from YAML `csi`.
        """
        in_csv = Path(in_csv) if in_csv is not None else (self.interim_dir / "satellite_irradiance.csv")
        if not in_csv.exists():
            logger.warning(f"[POST-SAT] Missing {in_csv}, skip.")
            return None

        out_csv = Path(out_csv) if out_csv is not None else (self.processed_dir / "satellite_irradiance_clean.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(in_csv)

        need = {self.col_time, self.col_lat, self.col_lon, self.csi_obs_col}
        missing = need - set(df.columns)
        if missing:
            logger.warning(f"[POST-SAT] Missing columns {missing} in {in_csv.name}, skip.")
            return None

        df[self.col_time] = _parse_time_col_to_utc_naive(df[self.col_time])
        df = df.sort_values(self.col_time)

        csi_col = self.csi_col
        clear_col = self.csi_clear_col

        clip_max = float(clip_max if clip_max is not None else self.csi_cfg.get("clip_max", 2.0))
        night_ghi_clear_min = float(
            night_ghi_clear_min if night_ghi_clear_min is not None else self.csi_cfg.get("ghi_clear_min", 20.0)
        )

        # 1) drop clipped CSI if present
        if csi_col in df.columns:
            df = _drop_csi_clipped(df, [csi_col], clip_max=clip_max)
        else:
            logger.warning(f"[POST-SAT] No CSI column '{csi_col}' -> clip filter skipped.")

        # index by time for resampling
        df = df.set_index(self.col_time)

        # numeric cast (avoid pandas FutureWarning)
        for c in df.columns:
            if c in (self.col_lat, self.col_lon):
                continue
            if df[c].dtype == "object":
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass

        # determine how many consecutive NaNs we can safely fill
        # e.g. original 30min -> target 10min => need to fill up to 2 missing points
        try:
            target_td = pd.to_timedelta(resample_rule)
            # median original delta per pixel varies a bit; use global median delta as heuristic
            orig_td = df.index.to_series().diff().median()
            if pd.isna(orig_td) or orig_td <= pd.Timedelta(0):
                fill_limit = 2
            else:
                ratio = int(round(orig_td / target_td)) - 1
                fill_limit = max(0, ratio)
                # keep it reasonable
                fill_limit = min(fill_limit, 12)
        except Exception:
            fill_limit = 2

        grouped = []
        for (lat, lon), g in df.groupby([self.col_lat, self.col_lon], sort=False):
            # 2) resample to dense grid
            g_rs = g.resample(resample_rule).mean(numeric_only=True)

            # --- FILL missing bins created by upsampling ---
            num_cols = g_rs.select_dtypes(include="number").columns.tolist()
            if num_cols:
                # Interpolate in time (fills internal gaps)
                if fill_limit > 0:
                    g_rs[num_cols] = g_rs[num_cols].interpolate(
                        method="time",
                        limit=fill_limit,
                        limit_direction="both",
                    )
                    # Small safety net at edges
                    g_rs[num_cols] = g_rs[num_cols].ffill(limit=fill_limit).bfill(limit=fill_limit)
                else:
                    # If we're downsampling or equal freq, no need to fill
                    pass
            # ---------------------------------------------

            g_rs[self.col_lat] = lat
            g_rs[self.col_lon] = lon
            grouped.append(g_rs.reset_index())

        df_rs = pd.concat(grouped, ignore_index=True)

        # 3) drop night via clear-sky column (robust to NaNs)
        if clear_col in df_rs.columns:
            df_rs = df_rs[df_rs[clear_col].fillna(-float("inf")) >= float(night_ghi_clear_min)]
        else:
            logger.warning(f"[POST-SAT] No clear-sky column '{clear_col}' -> night filter skipped.")

        df_rs.to_csv(out_csv, index=False)
        logger.success(f"[POST-SAT] Wrote {out_csv} (rows={len(df_rs)})")
        return out_csv


# ---------------------------------------------------------------------
# Dataset + DataLoader (same-day windows + no gaps)
# ---------------------------------------------------------------------
class ForecastDataset(Dataset):
    """
    Dataset windowing that NEVER crosses a day boundary
    and NEVER crosses a time gap.
    """

    def __init__(
        self,
        sat_data,
        ground_data,
        targets,
        timestamps,
        past_steps,
        future_steps,
        freq: str = "10min",
    ):
        self.sat = sat_data
        self.ground = ground_data
        self.targets = targets

        self.past = int(past_steps)
        self.future = int(future_steps)
        self.window = self.past + self.future

        self.timestamps = pd.to_datetime(pd.Index(timestamps))
        self.freq = pd.to_timedelta(freq)

        self.T = len(self.timestamps)

        self.valid_starts = self._compute_valid_starts()

    def _compute_valid_starts(self):
        t = self.timestamps
        dt = self.freq
        win = self.window

        valid = []
        for i in range(0, self.T - win + 1):
            seg = t[i : i + win]
            if not (seg.date == seg[0].date()).all():
                continue
            if not ((seg[1:] - seg[:-1]) == dt).all():
                continue
            valid.append(i)
        return np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        i = int(self.valid_starts[idx])

        X_sat = self.sat[i : i + self.past]
        if not torch.is_tensor(X_sat):
            X_sat = torch.tensor(X_sat, dtype=torch.float32)

        X_ground = self.ground[i : i + self.past]
        if not torch.is_tensor(X_ground):
            X_ground = torch.tensor(X_ground, dtype=torch.float32)

        # ADD FEATURE DIMENSION
        # (S, N) -> (S, N, 1)
        X_ground = X_ground.unsqueeze(-1)


        y = self.targets[i + self.past : i + self.past + self.future]
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)

        return {
            "satellite": X_sat,
            "ground": X_ground,
            "target": y,
        }
    

def make_dataloader(
    sat_data,
    ground_data,
    target_data,
    timestamps,
    cfg,
    batch_size=32,
    shuffle=False,
):
    past_steps = cfg["past_timesteps"]
    future_steps = cfg["future_timesteps"]
    freq = cfg.get("frequency", "10min")

    dataset = ForecastDataset(
        sat_data=sat_data,
        ground_data=ground_data,
        targets=target_data,
        timestamps=timestamps,
        past_steps=past_steps,
        future_steps=future_steps,
        freq=freq,
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

from pathlib import Path
import pandas as pd
import numpy as np
import requests

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from loguru import logger
import joblib
import yaml

class GroundPreprocessor:

    class Columns:
        def __init__(self, cfg):
            raw, renamed = cfg["columns"]["raw"], cfg["columns"]["renamed"]
            self.raw = raw
            self.renamed = renamed

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

    def __init__(self, cfg_path: Path, raw_dir: Path, interim_dir: Path, processed_dir: Path):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)
        self.meta_path = self.interim_dir / "ground_summary.csv"
        self.scaler = None

        self.cols = self.Columns(self.cfg)

        # MeteoSwiss 
        self.meta_url = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_stations.csv"
        self.station_url_tpl = (
            "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/"
            "{station_lower}/ogd-smn_{station_lower}_t_historical_2020-2029.csv"
        )

        # Target stations 
        self.test_stations = ["BIE", "PUY", "VEV", "ORO", "MLS", "MAH"]


    # Load metadata
    def load_station_metadata(self) -> pd.DataFrame:
        logger.info("Loading MeteoSwiss station metadata...")
        stations_df = pd.read_csv(self.meta_url, sep=";", encoding="latin1")
        logger.info(f"{len(stations_df)} stations found.")
        return stations_df

    # Download all test stations
    def download_all_stations(self) -> list[Path]:
        logger.info("Downloading ground station CSVs...")
        paths = []
        for code in self.test_stations:
            st = code.lower()
            url = self.station_url_tpl.format(station_lower=st)
            dest = self.raw_dir / f"{st}_historical.csv"

            if dest.exists():
                logger.debug(f"{code}: already downloaded.")
                paths.append(dest)
                continue

            resp = requests.get(url)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                logger.success(f"Downloaded {code}")
                paths.append(dest)
            elif resp.status_code == 404:
                logger.warning(f"No data found for {code}")
            else:
                logger.error(f"Error {resp.status_code} for {code}")
        return paths

    # Process and clean
    def process_station_file(self, input_path: Path) -> Path | None:
        try:
            df = pd.read_csv(input_path, sep=";")
            c = self.cols

            if c.timestamp not in df.columns:
                raise ValueError(f"Missing {c.timestamp} in {input_path.name}")

            # Convert time column
            df[c.r_timestamp] = pd.to_datetime(df[c.timestamp], format="%d.%m.%Y %H:%M", errors="coerce")

            # Rename columns
            if c.irradiance in df.columns:
                df = df.rename(columns={c.irradiance: c.r_irradiance})
            if c.station in df.columns:
                df = df.rename(columns={c.station: c.r_station})

            df = df.sort_values(c.r_timestamp).ffill().bfill()

            keep = [col for col in [c.r_timestamp, c.r_station, c.r_irradiance] if col in df.columns]
            df = df[keep]

            out = self.interim_dir / input_path.name.replace("_historical", "_processed")
            df.to_csv(out, index=False)
            logger.success(f"Processed {input_path.name} → {out.name}")
            return out
        except Exception as e:
            logger.error(f"Failed to process {input_path.name}: {e}")
            return None

    # Generate summary
    def generate_summary(self, processed_files: list[Path], stations_meta: pd.DataFrame):
        c = self.cols
        summary = []
        for f in processed_files:
            try:
                df = pd.read_csv(f)
                station = f.stem.split("_")[0].upper()
                meta_row = stations_meta.loc[stations_meta[c.station] == station]
                summary.append({
                    c.r_station: station,
                    "rows": len(df),
                    "start": df[c.r_timestamp].min(),
                    "end": df[c.r_timestamp].max(),
                    c.r_lat: meta_row[c.lat].values[0] if not meta_row.empty else None,
                    c.r_lon: meta_row[c.lon].values[0] if not meta_row.empty else None,
                    "altitude": meta_row[c.alt].values[0] if not meta_row.empty else None,
                })
            except Exception as e:
                logger.warning(f"Could not summarize {f.name}: {e}")

        if summary:
            pd.DataFrame(summary).to_csv(self.meta_path, index=False)
            logger.success("Ground summary saved.")
        else:
            logger.warning("No valid processed files found.")

    # ---------------------------------------------------------------------

    # Merge all stations
    def load_and_merge(self) -> pd.DataFrame:
        c = self.cols
        logger.info("Merging all processed ground data...")
        dfs = []
        for f in self.interim_dir.glob("*_processed.csv"):
            df = pd.read_csv(f)
            if {c.r_timestamp, c.r_station, c.r_irradiance}.issubset(df.columns):
                dfs.append(df)
        merged = pd.concat(dfs)
        pivot = merged.pivot(index=c.r_timestamp, columns=c.r_station, values=c.r_irradiance)
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index().interpolate().ffill().bfill()
        logger.success(f"Ground merged → {pivot.shape}")
        return pivot

    # # 6. Split chronologically
    # def split(self, df):
    #     n = len(df)
    #     tr, vr = self.cfg["splits"]["train_ratio"], self.cfg["splits"]["val_ratio"]
    #     return df.iloc[:int(n*tr)], df.iloc[int(n*tr):int(n*(tr+vr))], df.iloc[int(n*(tr+vr)):]

    # # 7. Scale
    # # ---------------------------------------------------------------------
    # def scale(self, train, val, test):
    #     self.scaler = StandardScaler().fit(train)
    #     joblib.dump(self.scaler, self.processed_dir / "scaler.joblib")
    #     logger.success("Scaler fitted and saved.")
    #     transform = lambda d: pd.DataFrame(self.scaler.transform(d), index=d.index, columns=d.columns)
    #     return transform(train), transform(val), transform(test)

    # # ---------------------------------------------------------------------
    # # 8. Build adjacency
    # # ---------------------------------------------------------------------
    # def build_adjacency(self):
    #     c = self.cols
    #     meta = pd.read_csv(self.meta_path)
    #     lat, lon = meta[c.r_lat], meta[c.r_lon]
    #     dist = cdist(np.c_[lat, lon], np.c_[lat, lon]) * 111  # km
    #     sigma, max_d = self.cfg["graph"]["sigma_km"], self.cfg["graph"]["max_dist_km"]
    #     A = np.exp(-dist**2 / (2*sigma**2))
    #     A[dist > max_d] = 0
    #     np.fill_diagonal(A, 0)
    #     return A

    # # ---------------------------------------------------------------------
    # # 9. Create sequences
    # # ---------------------------------------------------------------------
    # def create_sequences(self, X, window, horizon, stride):
    #     n_time, n_nodes = X.shape
    #     n_samples = (n_time - window - horizon) // stride + 1
    #     X_seq = np.lib.stride_tricks.sliding_window_view(X, (window, n_nodes))[::stride, 0]
    #     Y_seq = np.lib.stride_tricks.sliding_window_view(X, (horizon, n_nodes))[window::stride, 0]
    #     return np.expand_dims(X_seq, -1), np.expand_dims(Y_seq, -1)


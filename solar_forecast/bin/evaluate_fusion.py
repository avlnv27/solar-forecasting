from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from loguru import logger

from solar_forecast.train import load_model_cfg, prepare_data_and_graph
from solar_forecast.nn.utils import make_dataloader, ForecastDataset
from solar_forecast.nn.models.fusion import FusionModel
from solar_forecast.config.paths import PROCESSED_DATA_DIR


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def infer_freq_minutes(ts: pd.DatetimeIndex) -> int:
    dt = ts.to_series().diff().median()
    return int(dt.total_seconds() // 60)


def build_val_data(
    sat: torch.Tensor,
    ground: torch.Tensor,
    target: torch.Tensor,
    timestamps: pd.DatetimeIndex,
    val_ratio: float,
):
    n = len(target)
    n_train = int(n * (1.0 - val_ratio))
    return (
        sat[n_train:],
        ground[n_train:],
        target[n_train:],
        timestamps[n_train:],
    )


def load_model(
    cfg: dict,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    ckpt_path: Path,
    device: torch.device,
) -> FusionModel:
    model = FusionModel(
        cfg=cfg,
        cfg_sat=cfg["model"]["satellite"],
        cfg_ground=cfg["model"]["ground"],
        cfg_fusion=cfg["model"]["fusion"],
        A_ground=(edge_index, edge_weight),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    logger.success(f"Loaded model from {ckpt_path}")
    return model


@torch.no_grad()
def predict_multi_horizon(
    model: FusionModel,
    loader,
    device,
    edge_index,
    edge_weight,
) -> Tuple[np.ndarray, np.ndarray]:
    preds, trues = [], []

    for batch in loader:
        yhat = model(
            batch["satellite"].to(device),
            batch["ground"].to(device),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        preds.append(yhat.cpu())
        trues.append(batch["target"].cpu())

    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


def csi_to_ghi(csi: np.ndarray, ghi_clear: np.ndarray) -> np.ndarray:
    return csi * ghi_clear


def compute_metrics_per_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons_min: list[int],
) -> pd.DataFrame:
    rows = []
    for h in range(y_true.shape[1]):
        err = y_pred[:, h] - y_true[:, h]
        rows.append(
            dict(
                horizon_min=horizons_min[h],
                RMSE=np.sqrt(np.mean(err ** 2)),
                MAE=np.mean(np.abs(err)),
                MBE=np.mean(err),
            )
        )
    return pd.DataFrame(rows)


def plot_metric(df: pd.DataFrame, metric: str, out: Path):
    plt.figure()
    plt.plot(df["horizon_min"], df[metric], marker="o")
    plt.xlabel("Forecast horizon (min)")
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_forecast_timeseries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex,
    valid_starts: np.ndarray,
    past: int,
    horizon: int,
    out_path: Path,
    max_points: int = 500,
):
    times, yt, yp = [], [], []

    for k, i in enumerate(valid_starts):
        t_idx = i + past + horizon
        if t_idx >= len(timestamps):
            continue
        times.append(timestamps[t_idx])
        yt.append(y_true[k, horizon])
        yp.append(y_pred[k, horizon])

    times = np.array(times)
    yt = np.array(yt)
    yp = np.array(yp)

    if len(times) > max_points:
        idx = np.linspace(0, len(times) - 1, max_points).astype(int)
        times, yt, yp = times[idx], yt[idx], yp[idx]

    plt.figure(figsize=(10, 4))
    plt.plot(times, yt, label="Observed GHI", linewidth=2)
    plt.plot(times, yp, label="Forecast GHI", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("GHI [W/m²]")
    plt.title(f"GHI forecast vs observation ({(horizon+1)*10} min ahead)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def run_evaluation(
    ckpt_path: Path = PROCESSED_DATA_DIR / "checkpoints" / "fusion_best.pt",
    val_ratio: float = 0.2,
    batch_size: int = 32,
    out_dir: Path = PROCESSED_DATA_DIR / "evaluation",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = load_model_cfg()

    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    sat_val, ground_val, target_val, ts_val = build_val_data(
        sat, ground, target, timestamps, val_ratio
    )

    freq_min = infer_freq_minutes(ts_val)
    freq_str = f"{freq_min}min"

    past = cfg["past_timesteps"]
    future = cfg["future_timesteps"]

    val_dataset = ForecastDataset(
        sat_data=sat_val,
        ground_data=ground_val,
        targets=target_val,
        timestamps=ts_val,
        past_steps=past,
        future_steps=future,
        freq=freq_str,
    )

    val_loader = make_dataloader(
        sat_data=sat_val,
        ground_data=ground_val,
        target_data=target_val,
        timestamps=ts_val,
        cfg={**cfg, "frequency": freq_str},
        batch_size=batch_size,
        shuffle=False,
    )

    valid_starts = val_dataset.valid_starts
    logger.success(f"Validation windows: {len(valid_starts)}")

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = load_model(cfg, edge_index, edge_weight, ckpt_path, device)

    y_csi, yhat_csi = predict_multi_horizon(
        model, val_loader, device, edge_index, edge_weight
    )

    N, H = y_csi.shape

    # --- Clear-sky alignment (window level)
    ground_csv = next((PROCESSED_DATA_DIR / "ground").glob("*_clean.csv"))
    gdf = pd.read_csv(ground_csv)
    gdf["time"] = pd.to_datetime(gdf["time"])
    gdf = gdf.set_index("time")

    ghi_clear_series = gdf.loc[ts_val, "ghi_clear"].values

    ghi_clear = np.zeros((N, H), dtype=np.float32)
    for k, i in enumerate(valid_starts):
        ghi_clear[k, :] = ghi_clear_series[i + past : i + past + H]

    y_ghi = csi_to_ghi(y_csi, ghi_clear)
    yhat_ghi = csi_to_ghi(yhat_csi, ghi_clear)

    horizons_min = [(h + 1) * freq_min for h in range(H)]

    metrics = compute_metrics_per_horizon(y_ghi, yhat_ghi, horizons_min)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics_per_horizon.csv", index=False)

    plot_metric(metrics, "RMSE", out_dir / "rmse_vs_horizon.png")
    plot_metric(metrics, "MAE", out_dir / "mae_vs_horizon.png")
    plot_metric(metrics, "MBE", out_dir / "mbe_vs_horizon.png")

    plot_forecast_timeseries(
        y_ghi, yhat_ghi, ts_val, valid_starts, past, 0,
        out_dir / "forecast_10min.png"
    )
    plot_forecast_timeseries(
        y_ghi, yhat_ghi, ts_val, valid_starts, past, 2,
        out_dir / "forecast_30min.png"
    )
    plot_forecast_timeseries(
        y_ghi, yhat_ghi, ts_val, valid_starts, past, 5,
        out_dir / "forecast_60min.png"
    )

    logger.success(f"Evaluation complete → {out_dir}")


if __name__ == "__main__":
    run_evaluation()

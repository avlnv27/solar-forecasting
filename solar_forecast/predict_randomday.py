from __future__ import annotations

from pathlib import Path
from typing import Tuple
import random

import numpy as np
import pandas as pd
import torch
from loguru import logger

from solar_forecast.train import load_model_cfg, prepare_data_and_graph
from solar_forecast.nn.utils import make_dataloader, ForecastDataset
from solar_forecast.nn.models.fusion import FusionModel
from solar_forecast.nn.models.satellite_only import SatelliteOnlyModel
from solar_forecast.nn.models.ground_only import GroundOnlyModel
from solar_forecast.config.paths import PROCESSED_DATA_DIR


# ============================================================
# CONFIG – change only these if needed
# ============================================================

PREDICTION_OUTPUT_FILES = {
    "persistence":    "persistence_predictions.csv",
    "satellite_only": "satellite_only_predictions.csv",
    "ground_only":    "ground_only_predictions.csv",
    "fusion":         "fusion_predictions.csv",
}

CKPT_NAMES = {
    "satellite_only": "satellite_only_best.pt",
    "ground_only":    "ground_only_best.pt",
    "fusion":         "fusion_best_randday_20260202_201323.pt",
}

RANDOM_SEED = 42   # IMPORTANT: keep same as training if possible


# ============================================================
# HELPERS
# ============================================================

def infer_freq_minutes(ts: pd.DatetimeIndex) -> int:
    return int(ts.to_series().diff().median().total_seconds() // 60)


def build_val_data_random_days(
    sat: torch.Tensor,
    ground: torch.Tensor,
    target: torch.Tensor,
    timestamps: pd.DatetimeIndex,
    val_ratio: float,
    seed: int = 42,
):
    """
    Random-day split:
    - sample full days randomly
    - keep all timesteps from those days
    """
    rng = random.Random(seed)

    days = pd.to_datetime(timestamps.date)
    unique_days = sorted(days.unique())

    n_val_days = int(len(unique_days) * val_ratio)
    val_days = set(rng.sample(list(unique_days), n_val_days))

    val_mask = np.array([d in val_days for d in days])

    logger.info(
        f"Random-day split: {len(unique_days)} days total → {n_val_days} validation days"
    )

    return (
        sat[val_mask],
        ground[val_mask],
        target[val_mask],
        timestamps[val_mask],
    )


def csi_to_ghi(csi: np.ndarray, ghi_clear: np.ndarray) -> np.ndarray:
    return csi * ghi_clear


# ============================================================
# MODEL HELPERS
# ============================================================

def predict_persistence(
    target: torch.Tensor,
    valid_starts: list[int],
    past: int,
    future: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Persistence baseline:
    ŷ(t+h) = y(t)
    """
    y_true, y_pred = [], []

    for i in valid_starts:
        y0 = target[i + past - 1]
        y_true.append(target[i + past : i + past + future])
        y_pred.append(torch.full((future,), y0))

    return torch.stack(y_true).numpy(), torch.stack(y_pred).numpy()


def load_model(model_type, cfg, edge_index, edge_weight, ckpt_path, device):
    if model_type == "fusion":
        model = FusionModel(
            cfg=cfg,
            cfg_sat=cfg["model"]["satellite"],
            cfg_ground=cfg["model"]["ground"],
            cfg_fusion=cfg["model"]["fusion"],
            A_ground=(edge_index, edge_weight),
        )
    elif model_type == "satellite_only":
        model = SatelliteOnlyModel(cfg=cfg, cfg_sat=cfg["model"]["satellite"])
    elif model_type == "ground_only":
        model = GroundOnlyModel(
            cfg=cfg,
            cfg_ground=cfg["model"]["ground"],
            A_ground=(edge_index, edge_weight),
        )
    else:
        raise ValueError(model_type)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    logger.success(f"Loaded {model_type} model from {ckpt_path}")
    return model


@torch.no_grad()
def predict_multi_horizon(model, loader, device, edge_index, edge_weight):
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


# ============================================================
# MAIN PREDICT
# ============================================================

def run_predict(
    model_type: str,
    ckpt_path: Path | None,
    val_ratio: float = 0.2,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{model_type}] device={device}")

    cfg = load_model_cfg()
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    # ---- RANDOM DAY SPLIT (same logic as training)
    sat_v, ground_v, target_v, ts_v = build_val_data_random_days(
        sat, ground, target, timestamps, val_ratio, RANDOM_SEED
    )

    freq_min = infer_freq_minutes(ts_v)
    past, future = cfg["past_timesteps"], cfg["future_timesteps"]

    dataset = ForecastDataset(
        sat_data=sat_v,
        ground_data=ground_v,
        targets=target_v,
        timestamps=ts_v,
        past_steps=past,
        future_steps=future,
        freq=f"{freq_min}min",
    )

    loader = make_dataloader(
        sat_data=sat_v,
        ground_data=ground_v,
        target_data=target_v,
        timestamps=ts_v,
        cfg={**cfg, "frequency": f"{freq_min}min"},
        batch_size=batch_size,
        shuffle=False,
    )

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    if model_type == "persistence":
        y_csi, yhat_csi = predict_persistence(
            target_v, dataset.valid_starts, past, future
        )
    else:
        model = load_model(
            model_type, cfg, edge_index, edge_weight, ckpt_path, device
        )
        y_csi, yhat_csi = predict_multi_horizon(
            model, loader, device, edge_index, edge_weight
        )

    # ============================================================
    # CSI → GHI reconversion
    # ============================================================

    ground_csv = next((PROCESSED_DATA_DIR / "ground").glob("*_clean.csv"))
    gdf = pd.read_csv(ground_csv)
    gdf["time"] = pd.to_datetime(gdf["time"])
    gdf = gdf.set_index("time")

    ghi_clear_series = gdf.loc[ts_v, "ghi_clear"].values

    N, H = y_csi.shape
    ghi_clear = np.zeros((N, H), dtype=np.float32)
    ts_out = np.empty((N, H), dtype="datetime64[ns]")

    for k, i in enumerate(dataset.valid_starts):
        ghi_clear[k] = ghi_clear_series[i + past : i + past + H]
        ts_out[k] = ts_v[i + past : i + past + H]

    y_ghi = csi_to_ghi(y_csi, ghi_clear)
    yhat_ghi = csi_to_ghi(yhat_csi, ghi_clear)

    # ============================================================
    # SAVE CSV (LONG FORMAT)
    # ============================================================

    rows = []
    for k in range(N):
        for h in range(H):
            rows.append(dict(
                model=model_type,
                window_id=k,
                horizon_min=(h + 1) * freq_min,
                timestamp=ts_out[k, h],

                y_true_csi=y_csi[k, h],
                y_pred_csi=yhat_csi[k, h],
                y_true_ghi=y_ghi[k, h],
                y_pred_ghi=yhat_ghi[k, h],
            ))

    out_dir = PROCESSED_DATA_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / PREDICTION_OUTPUT_FILES[model_type]

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.success(f"[{model_type}] Predictions saved → {out_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    CKPT_DIR = PROCESSED_DATA_DIR / "checkpoints"

    run_predict("persistence", None)
    run_predict("satellite_only", CKPT_DIR / CKPT_NAMES["satellite_only"])
    run_predict("ground_only",    CKPT_DIR / CKPT_NAMES["ground_only"])
    run_predict("fusion",         CKPT_DIR / CKPT_NAMES["fusion"])

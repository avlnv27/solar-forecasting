from __future__ import annotations

from pathlib import Path
from typing import Tuple

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
# HELPERS
# ============================================================
def infer_freq_minutes(ts: pd.DatetimeIndex) -> int:
    return int(ts.to_series().diff().median().total_seconds() // 60)


def build_val_data(
    sat: torch.Tensor,
    ground: torch.Tensor,
    target: torch.Tensor,
    timestamps: pd.DatetimeIndex,
    val_ratio: float,
):
    n = len(target)
    n_train = int(n * (1.0 - val_ratio))
    return sat[n_train:], ground[n_train:], target[n_train:], timestamps[n_train:]


def csi_to_ghi(csi: np.ndarray, ghi_clear: np.ndarray) -> np.ndarray:
    return csi * ghi_clear


# ============================================================
# MODEL LOADERS
# ============================================================
def load_model(model_type: str, cfg, edge_index, edge_weight, ckpt_path, device):
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
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    logger.success(f"Loaded {model_type} model from {ckpt_path}")
    return model


@torch.no_grad()
def predict_multi_horizon(
    model,
    loader,
    device,
    model_type,
    edge_index,
    edge_weight,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for all batches.
    
    All models accept the same signature for compatibility:
    model(x_sat, x_ground, edge_index=..., edge_weight=...)
    """
    preds, trues = [], []

    for batch in loader:
        # All models use the same signature (even if some args are ignored)
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
    ckpt_path: Path,
    val_ratio: float = 0.2,
    batch_size: int = 32,
    out_dir: Path = PROCESSED_DATA_DIR / "predictions",
):
    assert model_type in {"satellite_only", "ground_only", "fusion"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{model_type}] Using device: {device}")

    # ---- config & data
    cfg = load_model_cfg()
    sat, ground, target, timestamps, edge_index, edge_weight = prepare_data_and_graph(cfg)

    sat_v, ground_v, target_v, ts_v = build_val_data(
        sat, ground, target, timestamps, val_ratio
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

    valid_starts = dataset.valid_starts
    logger.success(f"[{model_type}] Prediction windows: {len(valid_starts)}")

    # ---- model
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    model = load_model(
        model_type, cfg, edge_index, edge_weight, ckpt_path, device
    )

    y_csi, yhat_csi = predict_multi_horizon(
        model,
        loader,
        device,
        model_type,
        edge_index,
        edge_weight,
    )

    N, H = y_csi.shape

    # ---- clear-sky alignment
    ground_csv = next((PROCESSED_DATA_DIR / "ground").glob("*_clean.csv"))
    gdf = pd.read_csv(ground_csv)
    gdf["time"] = pd.to_datetime(gdf["time"])
    gdf = gdf.set_index("time")

    ghi_clear_series = gdf.loc[ts_v, "ghi_clear"].values

    ghi_clear = np.zeros((N, H), dtype=np.float32)
    ts_out = np.empty((N, H), dtype="datetime64[ns]")

    for k, i in enumerate(valid_starts):
        ghi_clear[k] = ghi_clear_series[i + past : i + past + H]
        ts_out[k] = ts_v[i + past : i + past + H]

    y_ghi = csi_to_ghi(y_csi, ghi_clear)
    yhat_ghi = csi_to_ghi(yhat_csi, ghi_clear)

    # ---- SAVE CSV (LONG FORMAT)
    rows = []
    for k in range(N):
        for h in range(H):
            rows.append(
                dict(
                    model=model_type,
                    window_id=k,
                    horizon_min=(h + 1) * freq_min,
                    timestamp=ts_out[k, h],
                    y_true_csi=y_csi[k, h],
                    y_pred_csi=yhat_csi[k, h],
                    y_true_ghi=y_ghi[k, h],
                    y_pred_ghi=yhat_ghi[k, h],
                )
            )

    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_type}_predictions.csv"
    df.to_csv(csv_path, index=False)

    logger.success(f"[{model_type}] Predictions saved to {csv_path}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    CKPT_DIR = PROCESSED_DATA_DIR / "checkpoints"

    for model_type in ["satellite_only", "ground_only", "fusion"]:
        run_predict(
            model_type=model_type,
            ckpt_path=CKPT_DIR / f"{model_type}_best.pt",
        )
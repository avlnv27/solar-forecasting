"""
Train deep learning models for solar irradiance forecasting.

Supported architectures:
  - ground: TimeThenGraphIsoModel (GRU + GraphConv)
  - satellite: SatelliteCNN (CNN-based spatial model)
  - dual: DualSourceModel (fusion of ground + satellite)

Usage:
    python -m solar_forecast.modeling.train --model ground --epochs 50 --lr 1e-3
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from loguru import logger
from tqdm import tqdm
import typer

# --- Project imports ---
from solar_forecast.config.paths import MODELS_DIR, PROCESSED_DATA_DIR
from solar_forecast.nn.models.time_then_graph_iso import TimeThenGraphIsoModel
from solar_forecast.nn.models.satellite_cnn import SatelliteCNN
from solar_forecast.nn.models.dual_source_model import DualSourceModel

app = typer.Typer()


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def load_numpy_tensor(path: Path, name: str) -> torch.Tensor:
    file = path / name
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")
    arr = np.load(file)
    return torch.tensor(arr, dtype=torch.float32)


def make_dataloader(X, Y, batch_size: int = 32, shuffle: bool = True):
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs=50):
    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X, Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            # Forward pass
            if isinstance(X, (tuple, list)):  # for dual model (Xg, A, Xs)
                y_pred = model(*X)
            else:
                y_pred = model(X)
            loss = criterion(y_pred, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                if isinstance(X, (tuple, list)):
                    y_pred = model(*X)
                else:
                    y_pred = model(X)
                val_loss += criterion(y_pred, Y).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        logger.info(f"[{epoch+1}/{epochs}] Train={avg_train:.4f}  Val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            out_path = MODELS_DIR / f"best_{model.__class__.__name__}.pt"
            torch.save(model.state_dict(), out_path)
            logger.success(f"New best model saved ({out_path.name}, val_loss={avg_val:.4f})")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
@app.command()
def main(
    model: str = typer.Option("ground", help="Model type: ground, satellite, or dual."),
    epochs: int = typer.Option(30, help="Number of training epochs."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    batch_size: int = typer.Option(32, help="Batch size."),
    processed_dir: Path = PROCESSED_DATA_DIR,
    model_dir: Path = MODELS_DIR,
):
    """
    Train a deep learning model for solar irradiance forecasting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Starting training for model='{model}' on device={device}.")

    # ------------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------------
    if model == "ground":
        X_train = load_numpy_tensor(processed_dir, "X_train.npy")
        Y_train = load_numpy_tensor(processed_dir, "Y_train.npy")
        A = load_numpy_tensor(processed_dir, "A.npy")

        train_loader = make_dataloader(X_train, Y_train, batch_size)
        val_loader = train_loader  # TODO: separate val split later

        model_obj = TimeThenGraphIsoModel(
            input_size=1,
            horizon=Y_train.shape[1],
            n_nodes=X_train.shape[2],
        )
        criterion = nn.MSELoss()

    elif model == "satellite":
        X_train = load_numpy_tensor(processed_dir, "X_satellite_train.npy")
        Y_train = load_numpy_tensor(processed_dir, "Y_satellite_train.npy")

        train_loader = make_dataloader(X_train, Y_train, batch_size)
        val_loader = train_loader

        model_obj = SatelliteCNN(in_channels=X_train.shape[1])
        criterion = nn.MSELoss()

    elif model == "dual":
        Xg_train = load_numpy_tensor(processed_dir, "X_ground_train.npy")
        Xs_train = load_numpy_tensor(processed_dir, "X_satellite_train.npy")
        Y_train = load_numpy_tensor(processed_dir, "Y_train.npy")
        A = load_numpy_tensor(processed_dir, "A.npy")

        # Repeat adjacency matrix for all samples
        A_batch = A.unsqueeze(0).repeat(Xg_train.size(0), 1, 1)
        dataset = TensorDataset(Xg_train, A_batch, Xs_train, Y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = train_loader

        model_obj = DualSourceModel(
            input_size=1,
            horizon=Y_train.shape[1],
            n_nodes=Xg_train.shape[2],
            satellite_in_channels=Xs_train.shape[1],
        )
        criterion = nn.MSELoss()

    else:
        raise typer.BadParameter(f"Invalid model type: {model}")

    optimizer = optim.Adam(model_obj.parameters(), lr=lr)

    # ------------------------------------------------------
    # 2. Train model
    # ------------------------------------------------------
    train_loop(model_obj, train_loader, val_loader, optimizer, criterion, device, epochs)

    logger.success(f"Training for model '{model}' complete. Saved to {model_dir / 'best_model.pt'}")


if __name__ == "__main__":
    app()

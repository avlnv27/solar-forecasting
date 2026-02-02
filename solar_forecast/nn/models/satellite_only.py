import torch
import torch.nn as nn

from solar_forecast.nn.models.satellite_cnn import SatelliteCNN


class SatelliteOnlyModel(nn.Module):
    """
    Satellite-only baseline using CNN on satellite patches.
    """

    def __init__(self, cfg, cfg_sat):
        super().__init__()

        self.horizon = cfg["future_timesteps"]

        self.encoder = SatelliteCNN(
            input_channels=cfg_sat["input_channels"],
            conv1_channels=cfg_sat["conv1_channels"],
            conv2_channels=cfg_sat["conv2_channels"],
            kernel_size=cfg_sat["kernel_size"],
            pool_size=cfg_sat["pool_size"],
            feature_dim=cfg_sat["feature_dim"],
        )

        self.head = nn.Linear(cfg_sat["feature_dim"], self.horizon)

    def forward(
        self,
        x_sat,
        x_ground=None,        # ignored
        edge_index=None,      # ignored
        edge_weight=None,     # ignored
    ):
        feat = self.encoder(x_sat)   # (B, d_sat)
        y = self.head(feat)           # (B, horizon)
        return y

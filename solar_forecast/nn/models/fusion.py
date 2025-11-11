import torch
import torch.nn as nn

from solar_forecast.nn.models.satellite_cnn import SatelliteCNN
from solar_forecast.nn.models.time_then_graph_iso import TimeThenGraphIsoModel


class FusionModel(nn.Module):
    """
    Combines a Satellite CNN encoder and a Ground GNN encoder
    using a fusion (dense) layer for final irradiance forecast.
    """
    def __init__(self, cfg_sat, cfg_ground, n_ground_nodes, A_ground=None):
        super().__init__()

        # Satellite branch
        self.sat_encoder = SatelliteCNN(
            input_channels=cfg_sat.get("input_channels", 8 * 13),
            feature_dim=cfg_sat.get("feature_dim", 128)
        )

        # Ground branch
        self.ground_encoder = TimeThenGraphIsoModel(
            input_size=cfg_ground.get("input_size", 1),
            horizon=cfg_ground.get("horizon", 12),
            n_nodes=n_ground_nodes,
            hidden_size=cfg_ground.get("hidden_size", 64),
            output_size=cfg_ground.get("output_size", 64)
        )

        # Fusion dense layer
        fusion_in = cfg_sat.get("feature_dim", 128) + cfg_ground.get("output_size", 64)
        hidden_dim = cfg_ground.get("fusion_hidden_dim", 128)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.A_ground = A_ground

    def forward(self, x_sat, x_ground, edge_index=None, edge_weight=None):
        """
        x_sat: (B, C, H, W)
        x_ground: (B, T, N, F)
        """
        sat_feat = self.sat_encoder(x_sat)        # (B, feature_dim_sat)
        ground_feat = self.ground_encoder(x_ground, edge_index, edge_weight)  # (B, feature_dim_ground)

        combined = torch.cat([sat_feat, ground_feat], dim=-1)
        return self.fusion(combined)

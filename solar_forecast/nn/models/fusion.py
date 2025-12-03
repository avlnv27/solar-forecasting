import torch
import torch.nn as nn

from solar_forecast.nn.models.satellite_cnn import SatelliteCNN
from solar_forecast.nn.models.time_then_graph_iso import TimeThenGraphIsoModel


# --------------------------------------------------------------------
# AJOUT : Attention pooling TRÈS SIMPLE sur les noeuds (N -> 1 vecteur)
# --------------------------------------------------------------------
class GraphAttentionPooling(nn.Module):
    def __init__(self, d_in, d_att=32):
        super().__init__()
        self.W = nn.Linear(d_in, d_att)
        self.v = nn.Linear(d_att, 1)

    def forward(self, x):
        # x: (B, N, d_in)
        h = torch.tanh(self.W(x))         # (B, N, d_att)
        e = self.v(h)                     # (B, N, 1)
        α = torch.softmax(e, dim=1)       # (B, N, 1)
        return (α * x).sum(dim=1)         # (B, d_in)



class FusionModel(nn.Module):
    """
    Combines a Satellite CNN encoder and a Ground GNN encoder
    using a fusion MLP for final irradiance forecast.

    Every argument is forwarded from YAML → FusionModel → Ground model.
    """
    def __init__(self, cfg, cfg_sat, cfg_ground, cfg_fusion, A_ground=None):
        super().__init__()

        self.horizon = cfg["future_timesteps"]

        # SATELLITE 
        self.sat_encoder = SatelliteCNN(
            input_channels=cfg_sat["input_channels"],
            conv1_channels=cfg_sat["conv1_channels"],
            conv2_channels=cfg_sat["conv2_channels"],
            kernel_size=cfg_sat["kernel_size"],
            pool_size=cfg_sat["pool_size"],
            feature_dim=cfg_sat["feature_dim"],
        )

        # GROUND 
        self.ground_encoder = TimeThenGraphIsoModel(
            input_size=cfg_ground["input_size"],
            horizon=cfg["future_timesteps"],
            n_nodes=cfg_ground["nodes"],
            hidden_size=cfg_ground["hidden_size"],
            output_size=cfg_ground["output_size"],

            # temporal GRU params
            time_layers=cfg_ground["time_layers"],
            time_skip_connect=cfg_ground["time_skip_connect"],

            # GNN parameters
            graph_layers=cfg_ground["graph_layers"],
            root_weight=cfg_ground["root_weight"],
            norm=cfg_ground["norm"],
            add_backward=cfg_ground["add_backward"],
            cached=cfg_ground["cached"],

            # embedding / misc
            emb_size=cfg_ground["emb_size"],
            add_embedding_before=cfg_ground["add_embedding_before"],
            use_local_weights=cfg_ground["use_local_weights"],
            activation=cfg_ground["activation"],
            noise_mode=cfg_ground["noise_mode"],

            exog_size=0,
        )

        # attention pooling to reduce (B, N, d_out) to (B, d_out)
        self.node_pool = GraphAttentionPooling(d_in=cfg_ground["output_size"])

        # ----------------- FUSION MLP HEAD -----------------
        d_sat = cfg_sat["feature_dim"]
        d_gnn = cfg_ground["output_size"]
        fusion_input_dim = d_sat + d_gnn

        hidden1 = cfg_fusion["hidden_dim"]
        hidden2 = cfg_fusion.get("hidden_dim2", hidden1)
        dropout_p = cfg_fusion.get("dropout", 0.1)

        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            # sortie: un vecteur de taille = horizon (multi-step)
            nn.Linear(hidden2, self.horizon)
        )

        # Option pour forcer l'irradiance >= 0
        self.out_activation = nn.ReLU()

        self.A_ground = A_ground

    def forward(self, x_sat, x_ground, edge_index=None, edge_weight=None):

        if edge_index is None:
            edge_index, edge_weight = self.A_ground

        # 1) features satellite: (B, d_sat)
        sat_feat = self.sat_encoder(x_sat)

        # 2) features sol (GNN): typiquement (B, H, N, d_gnn)
        ground_feat = self.ground_encoder(
            x_ground,
            edge_index=edge_index,
            edge_weight=edge_weight
        )

        # 3) pooling sur les nodes (et, si besoin, on a déjà pris le dernier H)
        local_feat = self.node_pool(ground_feat)  # (B, d_gnn)

        # 4) concat features
        fusion_in = torch.cat([sat_feat, local_feat], dim=-1)  # (B, d_sat + d_gnn)

        # 5) MLP pour prédire l'irradiance future
        y = self.mlp(fusion_in)       # (B, horizon)

        # 6) clamp pour éviter des valeurs négatives (optionnel mais pratique)
        y = self.out_activation(y)    # (B, horizon), >= 0

        return y
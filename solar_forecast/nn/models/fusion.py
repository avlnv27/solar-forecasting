import torch
import torch.nn as nn

from solar_forecast.nn.models.satellite_cnn import SatelliteCNN
from solar_forecast.nn.models.time_then_graph_iso import TimeThenGraphIsoModel


# ============================================================
# ATTENTION MODULES
# ============================================================

class TemporalAttention(nn.Module):
    """
    Attention over temporal dimension (paper-inspired).
    Input:  (B, T, D)
    Output: (B, D)
    """
    def __init__(self, d_in: int, d_att: int = 32):
        super().__init__()
        self.W = nn.Linear(d_in, d_att)
        self.v = nn.Linear(d_att, 1)

    def forward(self, x):
        h = torch.tanh(self.W(x))        # (B, T, d_att)
        e = self.v(h)                    # (B, T, 1)
        alpha = torch.softmax(e, dim=1)  # (B, T, 1)
        return (alpha * x).sum(dim=1)    # (B, D)


class GraphAttentionPooling(nn.Module):
    """
    Attention pooling over graph nodes.
    Input:  (B, N, D)
    Output: (B, D)
    """
    def __init__(self, d_in: int, d_att: int = 32):
        super().__init__()
        self.W = nn.Linear(d_in, d_att)
        self.v = nn.Linear(d_att, 1)

    def forward(self, x):
        h = torch.tanh(self.W(x))        # (B, N, d_att)
        e = self.v(h)                    # (B, N, 1)
        alpha = torch.softmax(e, dim=1)  # (B, N, 1)
        return (alpha * x).sum(dim=1)    # (B, D)


# ============================================================
# FUSION MODEL
# ============================================================

class FusionModel(nn.Module):
    """
    Satellite + Ground fusion model.

    Satellite branch:
        CNN (spatial) → GRU (temporal) → Temporal Attention

    Ground branch:
        TimeThenGraphIsoModel → Node Attention Pooling

    Fusion:
        concat → MLP (paper-inspired, 3 dense layers)
    """

    def __init__(self, cfg, cfg_sat, cfg_ground, cfg_fusion, A_ground=None):
        super().__init__()

        self.horizon = cfg["future_timesteps"]
        self.past_steps = cfg["past_timesteps"]
        self.use_sat_temporal = cfg_sat.get("use_temporal_encoder", True)

        # ====================================================
        # SATELLITE BRANCH
        # ====================================================

        sat_input_channels = 1 if self.use_sat_temporal else cfg_sat["input_channels"]

        self.sat_spatial_encoder = SatelliteCNN(
            input_channels=sat_input_channels,
            conv1_channels=cfg_sat["conv1_channels"],
            conv2_channels=cfg_sat["conv2_channels"],
            kernel_size=cfg_sat["kernel_size"],
            pool_size=cfg_sat["pool_size"],
            feature_dim=cfg_sat["feature_dim"],
        )

        if self.use_sat_temporal:
            self.sat_temporal_encoder = nn.GRU(
                input_size=cfg_sat["feature_dim"],
                hidden_size=cfg_sat["feature_dim"],
                num_layers=cfg_sat.get("temporal_layers", 2),
                batch_first=True,
                dropout=0.2 if cfg_sat.get("temporal_layers", 2) > 1 else 0.0,
            )

            # 🔴 TEMPORAL ATTENTION (paper-style)
            self.sat_temporal_attn = TemporalAttention(
                d_in=cfg_sat["feature_dim"],
                d_att=cfg_sat.get("attn_dim", 32),
            )

        # ====================================================
        # GROUND BRANCH
        # ====================================================

        self.ground_encoder = TimeThenGraphIsoModel(
            input_size=cfg_ground["input_size"],
            horizon=self.horizon,
            n_nodes=cfg_ground["nodes"],
            hidden_size=cfg_ground["hidden_size"],
            output_size=cfg_ground["output_size"],

            time_layers=cfg_ground["time_layers"],
            time_skip_connect=cfg_ground["time_skip_connect"],

            graph_layers=cfg_ground["graph_layers"],
            root_weight=cfg_ground["root_weight"],
            norm=cfg_ground["norm"],
            add_backward=cfg_ground["add_backward"],
            cached=cfg_ground["cached"],

            emb_size=cfg_ground["emb_size"],
            add_embedding_before=cfg_ground["add_embedding_before"],
            use_local_weights=cfg_ground["use_local_weights"],
            activation=cfg_ground["activation"],
            noise_mode=cfg_ground["noise_mode"],

            exog_size=0,
        )

        self.node_pool = GraphAttentionPooling(
            d_in=cfg_ground["output_size"]
        )

        # ====================================================
        # FUSION MLP (paper-inspired)
        # ====================================================

        d_sat = cfg_sat["feature_dim"]
        d_gnn = cfg_ground["output_size"]
        fusion_input_dim = d_sat + d_gnn

        x1 = cfg_fusion.get("x1", 64)
        x2 = cfg_fusion.get("x2", 32)
        x3 = cfg_fusion.get("x3", 16)

        act_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }

        act1 = act_map.get(cfg_fusion.get("activation1", "relu"))
        act2 = act_map.get(cfg_fusion.get("activation2", "gelu"))
        act3 = act_map.get(cfg_fusion.get("activation3", "relu"))

        dropout_p = cfg_fusion.get("dropout", 0.3)

        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, x1),
            act1,
            nn.Dropout(dropout_p),

            nn.Linear(x1, x2),
            act2,
            nn.Dropout(dropout_p),

            nn.Linear(x2, x3),
            act3,

            nn.Linear(x3, self.horizon),
        )

        self.A_ground = A_ground

    # ====================================================
    # FORWARD
    # ====================================================

    def forward(self, x_sat, x_ground, edge_index=None, edge_weight=None):
        if edge_index is None:
            if self.A_ground is None:
                raise ValueError("edge_index missing and A_ground not set.")
            edge_index, edge_weight = self.A_ground

        # ================= SATELLITE =================
        if self.use_sat_temporal:
            B, T, H, W = x_sat.shape

            sat_feats = []
            for t in range(T):
                img_t = x_sat[:, t:t+1, :, :]
                feat_t = self.sat_spatial_encoder(img_t)
                sat_feats.append(feat_t)

            sat_feats = torch.stack(sat_feats, dim=1)  # (B, T, D)

            sat_feats, _ = self.sat_temporal_encoder(sat_feats)

            # ATTENTION instead of last timestep
            sat_feat = self.sat_temporal_attn(sat_feats)  # (B, D)

        else:
            sat_feat = self.sat_spatial_encoder(x_sat)

        # ================= GROUND =================
        ground_feat = self.ground_encoder(
            x_ground,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        if ground_feat.dim() == 4:
            ground_last = ground_feat[:, -1]  # (B, N, D)
        elif ground_feat.dim() == 3:
            ground_last = ground_feat
        else:
            raise RuntimeError(f"Unexpected ground_feat shape {ground_feat.shape}")

        local_feat = self.node_pool(ground_last)  # (B, D)

        # ================= FUSION =================
        fusion_in = torch.cat([sat_feat, local_feat], dim=-1)
        y = self.mlp(fusion_in)

        return y

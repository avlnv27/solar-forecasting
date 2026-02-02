import torch
import torch.nn as nn

from solar_forecast.nn.models.satellite_cnn import SatelliteCNN
from solar_forecast.nn.models.time_then_graph_iso import TimeThenGraphIsoModel


class GraphAttentionPooling(nn.Module):
    def __init__(self, d_in, d_att=32):
        super().__init__()
        self.W = nn.Linear(d_in, d_att)
        self.v = nn.Linear(d_att, 1)

    def forward(self, x):
        """
        x: (B, N, d_in)
        """
        h = torch.tanh(self.W(x))      # (B, N, d_att)
        e = self.v(h)                  # (B, N, 1)
        alpha = torch.softmax(e, dim=1)  # (B, N, 1)
        return (alpha * x).sum(dim=1)  # (B, d_in)


class FusionModel(nn.Module):
    """
    Combines SatelliteCNN and TimeThenGraphIsoModel
    using a fusion MLP for final irradiance forecast.
    
    IMPROVED VERSION: Adds temporal encoder (GRU) for satellite branch
    to match the temporal modeling capability of the ground branch.
    """
    def __init__(self, cfg, cfg_sat, cfg_ground, cfg_fusion, A_ground=None):
        super().__init__()

        self.horizon = cfg["future_timesteps"]
        self.past_steps = cfg["past_timesteps"]
        
        # Flag to use temporal encoder or not (for backward compatibility)
        self.use_sat_temporal = cfg_sat.get("use_temporal_encoder", True)

        # SATELLITE SPATIAL ENCODER 
        # Changed: input_channels = 1 (process one image at a time)
        # instead of input_channels = past_timesteps (treating time as channels)
        sat_input_channels = 1 if self.use_sat_temporal else cfg_sat["input_channels"]
        
        self.sat_spatial_encoder = SatelliteCNN(
            input_channels=sat_input_channels,
            conv1_channels=cfg_sat["conv1_channels"],
            conv2_channels=cfg_sat["conv2_channels"],
            kernel_size=cfg_sat["kernel_size"],
            pool_size=cfg_sat["pool_size"],
            feature_dim=cfg_sat["feature_dim"],
        )
        
        # SATELLITE TEMPORAL ENCODER (NEW!)
        if self.use_sat_temporal:
            self.sat_temporal_encoder = nn.GRU(
                input_size=cfg_sat["feature_dim"],
                hidden_size=cfg_sat["feature_dim"],
                num_layers=cfg_sat.get("temporal_layers", 2),
                batch_first=True,
                dropout=0.1 if cfg_sat.get("temporal_layers", 2) > 1 else 0.0
            )

        # GROUND ENCODER (Time-Then-Graph) - UNCHANGED
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

        # Attention pooling on ground nodes to get 1 node 
        self.node_pool = GraphAttentionPooling(d_in=cfg_ground["output_size"])

        # FUSION 
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

            nn.Linear(hidden2, self.horizon),  # (B, horizon)
        )

        self.A_ground = A_ground

    def forward(self, x_sat, x_ground, edge_index=None, edge_weight=None):
        # Get graph structure if not provided
        if edge_index is None:
            if self.A_ground is None:
                raise ValueError("Either edge_index must be provided in forward "
                                 "or A_ground must be set in the model.")
            edge_index, edge_weight = self.A_ground

        # === SATELLITE BRANCH ===
        if self.use_sat_temporal:
            # x_sat: (B, T, H, W) where T = past_timesteps
            B, T, H, W = x_sat.shape
            
            # Process each timestep separately with spatial CNN
            sat_feats = []
            for t in range(T):
                # Extract timestep t and add channel dimension
                img_t = x_sat[:, t:t+1, :, :]  # (B, 1, H, W)
                feat_t = self.sat_spatial_encoder(img_t)  # (B, feature_dim)
                sat_feats.append(feat_t)
            
            sat_feats = torch.stack(sat_feats, dim=1)  # (B, T, feature_dim)
            
            # Temporal encoding with GRU
            sat_feats, _ = self.sat_temporal_encoder(sat_feats)  # (B, T, feature_dim)
            sat_feat = sat_feats[:, -1]  # Take last timestep (B, feature_dim)
        else:
            # Old behavior: treat timesteps as channels
            # x_sat: (B, T, H, W) interpreted as (B, C=T, H, W)
            sat_feat = self.sat_spatial_encoder(x_sat)  # (B, feature_dim)

        # === GROUND BRANCH === (UNCHANGED)
        ground_feat = self.ground_encoder(
            x_ground,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        
        # (B, H, N, d_gnn) -> take last horizon
        if ground_feat.dim() == 4:
            ground_last = ground_feat[:, -1]      # (B, N, d_gnn)
        elif ground_feat.dim() == 3:
            ground_last = ground_feat # (B, N, d_gnn)
        else:
            raise RuntimeError(f"Unexpected ground_feat shape: {ground_feat.shape}")

        # Pooling nodes (B, d_gnn)
        local_feat = self.node_pool(ground_last)

        # === FUSION ===
        fusion_in = torch.cat([sat_feat, local_feat], dim=-1)  # (B, d_sat + d_gnn)
        y = self.mlp(fusion_in)        # (B, horizon)

        return y
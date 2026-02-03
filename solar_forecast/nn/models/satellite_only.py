import torch
import torch.nn as nn

from solar_forecast.nn.models.satellite_cnn import SatelliteCNN


class TemporalAttention(nn.Module):
    """
    Attention over temporal dimension.
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


class SatelliteOnlyModel(nn.Module):
    """
    Satellite-only baseline using CNN on satellite patches.
    
    IMPROVED VERSION:
    - Temporal encoder (GRU) to model sequential nature
    - Temporal attention (instead of just last timestep)
    - Batched timestep processing (faster!)
    """

    def __init__(self, cfg, cfg_sat):
        super().__init__()

        self.horizon = cfg["future_timesteps"]
        self.past_steps = cfg["past_timesteps"]
        
        # Flag to use temporal encoder or not (for backward compatibility)
        self.use_temporal = cfg_sat.get("use_temporal_encoder", True)
        
        # SPATIAL ENCODER
        sat_input_channels = 1 if self.use_temporal else cfg_sat["input_channels"]
        
        self.spatial_encoder = SatelliteCNN(
            input_channels=sat_input_channels,
            conv1_channels=cfg_sat["conv1_channels"],
            conv2_channels=cfg_sat["conv2_channels"],
            kernel_size=cfg_sat["kernel_size"],
            pool_size=cfg_sat["pool_size"],
            feature_dim=cfg_sat["feature_dim"],
        )
        
        # TEMPORAL ENCODER
        if self.use_temporal:
            self.temporal_encoder = nn.GRU(
                input_size=cfg_sat["feature_dim"],
                hidden_size=cfg_sat["feature_dim"],
                num_layers=cfg_sat.get("temporal_layers", 2),
                batch_first=True,
                dropout=0.1 if cfg_sat.get("temporal_layers", 2) > 1 else 0.0
            )
            
            # FIXED: Add temporal attention (like in FusionModel!)
            self.temporal_attn = TemporalAttention(
                d_in=cfg_sat["feature_dim"],
                d_att=cfg_sat.get("attn_dim", 32)
            )
        
        # PREDICTION HEAD
        head_hidden = cfg_sat.get("head_hidden_dim", cfg_sat["feature_dim"] // 2)
        self.head = nn.Sequential(
            nn.Linear(cfg_sat["feature_dim"], head_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(head_hidden),
            nn.Linear(head_hidden, self.horizon)
        )

    def forward(
        self,
        x_sat,
        x_ground=None,        # ignored (for compatibility)
        edge_index=None,      # ignored (for compatibility)
        edge_weight=None,     # ignored (for compatibility)
    ):
        """
        Args:
            x_sat: (B, T, H, W) - satellite image sequence
            x_ground: ignored (for compatibility with train loop)
            edge_index: ignored (for compatibility)
            edge_weight: ignored (for compatibility)
        
        Returns:
            y: (B, horizon) - predictions
        """
        
        if self.use_temporal:
            # Check dimensions
            if x_sat.dim() == 3:  # (T, H, W) - missing batch dim
                x_sat = x_sat.unsqueeze(0)  # (1, T, H, W)
            
            if x_sat.dim() != 4:
                raise ValueError(f"Expected 4D tensor (B, T, H, W), got shape {x_sat.shape}")
            
            B, T, H, W = x_sat.shape
            
            # FIXED: Batch process all timesteps at once (MUCH FASTER!)
            x_sat_flat = x_sat.reshape(B * T, 1, H, W)  # Merge batch and time
            feats_flat = self.spatial_encoder(x_sat_flat)  # (B*T, feature_dim)
            feats = feats_flat.reshape(B, T, -1)  # (B, T, feature_dim)
            
            # Temporal encoding with GRU
            feats, _ = self.temporal_encoder(feats)  # (B, T, feature_dim)
            
            # FIXED: Use attention instead of just last timestep!
            feat = self.temporal_attn(feats)  # (B, feature_dim)
            
        else:
            # Old behavior: treat timesteps as channels
            feat = self.spatial_encoder(x_sat)  # (B, feature_dim)
        
        # Prediction
        y = self.head(feat)  # (B, horizon)
        
        return y
import torch
import torch.nn as nn

from solar_forecast.nn.models.satellite_cnn import SatelliteCNN


class SatelliteOnlyModel(nn.Module):
    """
    Satellite-only baseline using CNN on satellite patches.
    
    IMPROVED VERSION: Adds temporal encoder (GRU) to properly model
    the sequential nature of satellite image time series.
    """

    def __init__(self, cfg, cfg_sat):
        super().__init__()

        self.horizon = cfg["future_timesteps"]
        self.past_steps = cfg["past_timesteps"]
        
        # Flag to use temporal encoder or not (for backward compatibility)
        self.use_temporal = cfg_sat.get("use_temporal_encoder", True)
        
        # SPATIAL ENCODER
        # Changed: input_channels = 1 (process one image at a time)
        sat_input_channels = 1 if self.use_temporal else cfg_sat["input_channels"]
        
        self.spatial_encoder = SatelliteCNN(
            input_channels=sat_input_channels,
            conv1_channels=cfg_sat["conv1_channels"],
            conv2_channels=cfg_sat["conv2_channels"],
            kernel_size=cfg_sat["kernel_size"],
            pool_size=cfg_sat["pool_size"],
            feature_dim=cfg_sat["feature_dim"],
        )
        
        # TEMPORAL ENCODER (NEW!)
        if self.use_temporal:
            self.temporal_encoder = nn.GRU(
                input_size=cfg_sat["feature_dim"],
                hidden_size=cfg_sat["feature_dim"],
                num_layers=cfg_sat.get("temporal_layers", 2),
                batch_first=True,
                dropout=0.1 if cfg_sat.get("temporal_layers", 2) > 1 else 0.0
            )
        
        # PREDICTION HEAD
        head_hidden = cfg_sat.get("head_hidden_dim", cfg_sat["feature_dim"] // 2)
        self.head = nn.Sequential(
            nn.Linear(cfg_sat["feature_dim"], head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
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
            # x_sat should be (B, T, H, W) where T = past_timesteps
            
            # Check dimensions and get shape
            if x_sat.dim() == 3:  # (T, H, W) - missing batch dim
                x_sat = x_sat.unsqueeze(0)  # (1, T, H, W)
            
            # Now it should be 4D: (B, T, H, W)
            if x_sat.dim() != 4:
                raise ValueError(f"Expected 4D tensor (B, T, H, W), got shape {x_sat.shape}")
            
            B, T, H, W = x_sat.shape
            
            # Process each timestep separately with spatial CNN
            feats = []
            for t in range(T):
                # Extract timestep t and add channel dimension
                img_t = x_sat[:, t:t+1, :, :]  # (B, 1, H, W)
                feat_t = self.spatial_encoder(img_t)  # (B, feature_dim)
                feats.append(feat_t)
            
            feats = torch.stack(feats, dim=1)  # (B, T, feature_dim)
            
            # Temporal encoding with GRU
            feats, _ = self.temporal_encoder(feats)  # (B, T, feature_dim)
            feat = feats[:, -1]  # Take last timestep (B, feature_dim)
        else:
            # Old behavior: treat timesteps as channels
            # x_sat: (B, T, H, W) interpreted as (B, C=T, H, W)
            feat = self.spatial_encoder(x_sat)  # (B, feature_dim)
        
        # Prediction
        y = self.head(feat)  # (B, horizon)
        
        return y
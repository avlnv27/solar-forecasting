import torch
import torch.nn as nn

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
        h = torch.tanh(self.W(x))        # (B, N, d_att)
        e = self.v(h)                    # (B, N, 1)
        alpha = torch.softmax(e, dim=1)  # (B, N, 1)
        return (alpha * x).sum(dim=1)    # (B, d_in)

class GroundOnlyModel(nn.Module):
    """
    Ground-only baseline using Time-Then-Graph model.
    """

    def __init__(self, cfg, cfg_ground, A_ground=None):
        super().__init__()

        self.horizon = cfg["future_timesteps"]

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

        self.node_pool = GraphAttentionPooling(d_in=cfg_ground["output_size"])
        
        # Add dropout for better regularization
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(cfg_ground["output_size"], self.horizon)
        )

        self.A_ground = A_ground

    def forward(
        self,
        x_sat,           # ignored (for compatibility with train loop)
        x_ground,        # actual input
        edge_index=None,
        edge_weight=None,
    ):
        """
        Args:
            x_sat: ignored (for compatibility)
            x_ground: (B, T, N, F) - ground station data
            edge_index: graph edge indices
            edge_weight: graph edge weights
        
        Returns:
            y: (B, horizon) - predictions
        """
        if edge_index is None:
            if self.A_ground is None:
                raise ValueError("edge_index not provided and A_ground is None")
            edge_index, edge_weight = self.A_ground

        ground_feat = self.ground_encoder(
            x_ground,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

        # (B, H, N, d) or (B, N, d)
        if ground_feat.dim() == 4:
            ground_last = ground_feat[:, -1]
        else:
            ground_last = ground_feat

        pooled = self.node_pool(ground_last)
        y = self.head(pooled)

        return y
from typing import Literal, Union, List, Optional
import torch.nn as nn
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import GraphConv, DiffConv
from tsl.utils import ensure_list
from .prototypes import TimeThenSpace


def _build_norm_layer(norm_type: str, hidden_size: int) -> nn.Module:
    """Return the right nn.Module for the requested norm string."""
    if norm_type in ("none", "None", ""):
        return nn.Identity()
    elif norm_type == "batchnorm":
        return nn.BatchNorm1d(hidden_size)
    elif norm_type == "layernorm":
        return nn.LayerNorm(hidden_size)
    else:
        raise ValueError(
            f"Unsupported norm '{norm_type}'. "
            f"Use 'batchnorm', 'layernorm', or 'none'."
        )


class TimeThenGraphIsoModel(TimeThenSpace):

    def __init__(self,
                 input_size: int,
                 horizon: int,
                 n_nodes: int,
                 output_size: int,
                 hidden_size: int,
                 time_layers: int = 1,
                 graph_layers: int = 1,
                 root_weight: bool = True,
                 norm: str = "none",
                 add_backward: bool = False,
                 cached: bool = False,
                 emb_size: int = 0,
                 add_embedding_before: Union[str, List[str], None] = 'encoding',
                 use_local_weights: Union[str, List[str], None] = None,
                 exog_size: int = 0,
                 activation: str = "elu",
                 noise_mode: Literal["lin", "multi", "add", "none"] = "lin",
                 time_skip_connect: bool = False):

        # ── TEMPORAL (GRU) ──────────────────────────────────────────────
        rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=time_layers,
            return_only_last_state=True,
            cell='gru'
        )

        # ── SPATIAL ─────────────────────────────────────────────────────
        mp_kwargs = dict(root_weight=root_weight, activation=activation)
        if add_backward:
            mp_conv = DiffConv
            mp_kwargs.update(k=1, add_backward=True)
        else:
            mp_conv = GraphConv
            mp_kwargs.update(norm="mean", cached=cached)

        mp_layers = nn.ModuleList([
            mp_conv(
                hidden_size + (emb_size if add_embedding_before and
                               'message_passing' in ensure_list(add_embedding_before) else 0),
                hidden_size,
                **mp_kwargs
            )
            for _ in range(graph_layers)
        ])

        # ── PARENT INIT ─────────────────────────────────────────────────
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            temporal_encoder=rnn,
            spatial_encoder=mp_layers,
            n_nodes=n_nodes,
            output_size=output_size,
            exog_size=exog_size,
            hidden_size=hidden_size,
            emb_size=emb_size,
            add_embedding_before=add_embedding_before,
            use_local_weights=use_local_weights,
            activation=activation,
            noise_mode=noise_mode,
            time_skip_connect=time_skip_connect,
        )

        # ── NORM LAYERS — registered AFTER super().__init__() ──────────
        # This is now 100% safe: _modules dict exists, nothing will wipe it.
        self._norm_layers = nn.ModuleList([
            _build_norm_layer(norm, hidden_size)
            for _ in range(graph_layers)
        ])

    # ── stmp uses getattr so it survives if called before _norm_layers ──
    def stmp(self, x, edge_index, edge_weight=None, emb=None):
        from solar_forecast.nn.utils import maybe_cat_emb

        # TEMPORAL
        out = self.temporal_encoder(x)
        x_enc = out.clone()

        # Grab norm layers — falls back to a list of Identity() if not
        # yet assigned (i.e. if somehow called during super().__init__()).
        norm_layers = getattr(self, '_norm_layers', None)
        if norm_layers is None:
            norm_layers = [nn.Identity() for _ in self.mp_layers]

        # SPATIAL + NORM
        for layer, norm in zip(self.mp_layers, norm_layers):
            if "message_passing" in self.add_embedding_before and emb is not None:
                out = maybe_cat_emb(out, emb)

            out = layer(out, edge_index, edge_weight)   # (B, N, hidden)

            # BatchNorm1d needs (B*N, H); LayerNorm/Identity are fine with (B, N, H)
            if isinstance(norm, nn.BatchNorm1d):
                B, N, H = out.shape
                out = norm(out.reshape(B * N, H)).reshape(B, N, H)
            else:
                out = norm(out)

        # OPTIONAL temporal skip
        if self.skip_connect is not None:
            out = out + self.skip_connect(x_enc)

        return out
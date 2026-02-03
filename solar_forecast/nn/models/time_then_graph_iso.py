from typing import Literal, Union, List, Optional
import torch.nn as nn
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import GraphConv, DiffConv
from tsl.utils import ensure_list
from .prototypes import TimeThenSpace


class TimeThenGraphIsoModel(TimeThenSpace):
    """
    GRU + GraphConv model that accepts ALL YAML parameters.
    (Even if some are not used internally.)
    """

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

        # TEMPORAL (GRU) 
        rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=time_layers,
            return_only_last_state=True,
            cell='gru'
        )

        # SPATIAL (GraphConv or DiffConv) 
        # (B, N, d_hidden)
        mp_kwargs = dict(root_weight=root_weight, activation=activation)

        if add_backward:
            mp_conv = DiffConv
            mp_kwargs.update(k=1, add_backward=True)
        else:
            mp_conv = GraphConv
            mp_kwargs.update(norm=norm, cached=cached)

        mp_layers = nn.ModuleList([
            mp_conv(
                hidden_size + (emb_size if add_embedding_before and 'message_passing' in ensure_list(add_embedding_before) else 0),
                hidden_size,
                **mp_kwargs
            )
            for _ in range(graph_layers)
        ])

        # PARENT INIT (TimeThenSpace / STGNN) 
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

from typing import Dict, Union

import torch
from torch import nn
from torch_geometric.nn import HANConv

class HAN(nn.Module):
    def __init__(self, data,
                 target_node_type,
                 in_channels: Union[int, Dict[str, int]],
                 hidden_channels=128,
                 num_heads=8,
                 dropout_rate=0.):
        super().__init__()
        self.target_node_type = target_node_type
        node_offset = data.node_offsets[target_node_type]
        y = data.node_stores[node_offset].y
        out_channels = torch.max(y) - torch.min(y) + 1

        self.han_conv = HANConv(in_channels, hidden_channels, heads=num_heads,
                                dropout=dropout_rate, metadata=data.metadata())
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, return_semantic_attention_weights: bool = False):
        if return_semantic_attention_weights:
            out, semantic_attn_dict = self.han_conv(x_dict, edge_index_dict, True)
            embedding = out[self.target_node_type]
            out = self.lin(embedding)
            return embedding, out, semantic_attn_dict
        else:
            out = self.han_conv(x_dict, edge_index_dict)
            embedding = out[self.target_node_type]
            out = self.lin(embedding)
            return embedding, out



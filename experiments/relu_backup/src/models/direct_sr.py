import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GraphNorm


class DirectSR(nn.Module):
    def __init__(self, config=None, *, n_source_nodes=None,
                 n_target_nodes=None, model_config=None):
        """Use explicit graph sizes/model_config; config is the legacy adapter."""
        super().__init__()
        if config is not None:
            n_source_nodes = config.dataset.n_source_nodes
            n_target_nodes = config.dataset.n_target_nodes
            model_config = config.model

        num_heads = model_config.num_heads
        edge_dim = model_config.edge_dim
        dropout = model_config.dropout
        beta = model_config.beta

        assert n_target_nodes % num_heads == 0

        self.conv1 = TransformerConv(n_source_nodes, n_source_nodes,
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn1 = GraphNorm(num_heads * n_source_nodes)
        self.conv2 = TransformerConv(num_heads * n_source_nodes, n_target_nodes // num_heads,
                                     heads=num_heads, edge_dim=edge_dim,
                                     dropout=dropout, beta=beta)
        self.bn2 = GraphNorm(n_target_nodes)


    def forward(self, source_pyg, target_m):
        x, edge_index, edge_attr = source_pyg.x, source_pyg.pos_edge_index, source_pyg.edge_attr

        # Update node embeddings for the source graph
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)

        # Super-resolve source graph using matrix multiplication
        xt = x.T @ x

        # Normalize values to be between [0, 1] as xt is treated as the target adjacency matrix
        xt_min = torch.min(xt)
        xt_max = torch.max(xt)
        pred_m = (xt - xt_min) / (xt_max - xt_min + 1e-8)  # Add epsilon to avoid division by zero

        return pred_m, target_m

import torch
import torch.nn as nn
from torch_geometric.utils import (
    softmax
)
from torch_scatter import scatter


class GAT_AD(nn.Module):
    def __init__(
        self,
        num_paths,
        hidden_dimension,
        dropout,
        window_size,
        temperature=1
    ) -> None:
        super().__init__()
        self.num_paths = num_paths
        self.hidden_dimension = hidden_dimension
        self.dropout = dropout
        self.window_size = window_size
        self.temperature = temperature

        self.attention = nn.Sequential(
            nn.Linear(self.window_size * 2, self.hidden_dimension),
            nn.ReLU(),
            nn.Linear(self.hidden_dimension, 1)
        )  # our pseudo-attention

    def forward(
        self,
        data
    ):
        x = data.x.clone()
        y = data.y.clone()
        device = x.device
        edge_index = data.edge_index

        batch_num = data.batch.max() + 1
        num_nodes = torch.tensor(data.num_nodes).to(device)  # necessary for MPS training
        node_num = num_nodes // batch_num

        src_path_indexes, dst_path_indexes = edge_index[0], edge_index[1]

        alpha = self.attention(
            torch.cat(
                (
                    x[src_path_indexes],
                    x[dst_path_indexes]
                ),
                dim=1
            )
        )
        alpha = softmax(alpha / self.temperature, dst_path_indexes)

        y_hat = scatter(
            y[src_path_indexes].unsqueeze(1) * alpha,
            dst_path_indexes,
            dim=0,
            reduce="sum"
        )

        return y_hat.view(-1), alpha.detach()

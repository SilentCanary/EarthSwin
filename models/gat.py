import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.feature_projection = nn.Linear(
            input_dim,
            output_dim,
            bias=False
        )

        self.attention_source = nn.Linear(
            output_dim,
            1,
            bias=False
        )

        self.attention_target = nn.Linear(
            output_dim,
            1,
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, edge_index, edge_weights):
        # node_features: [N, input_dim]
        # edge_index: [2, E]
        # edge_weights: [E]

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        # --------------------------------------------------
        # 1. Project node features
        # --------------------------------------------------

        projected_features = self.feature_projection(
            node_features
        )

        # --------------------------------------------------
        # 2. Get source and target features for each edge
        # --------------------------------------------------

        source_features = projected_features[source_nodes]
        target_features = projected_features[target_nodes]

        # --------------------------------------------------
        # 3. Calculate attention scores
        # --------------------------------------------------

        source_attention = self.attention_source(
            source_features
        ).squeeze(-1)

        target_attention = self.attention_target(
            target_features
        ).squeeze(-1)

        attention_scores = (
            source_attention +
            target_attention
        )

        attention_scores = F.leaky_relu(
            attention_scores,
            negative_slope=0.2
        )

        # --------------------------------------------------
        # 4. Incorporate graph edge weights
        # --------------------------------------------------

        attention_scores = (
            attention_scores +
            torch.log(edge_weights + 1e-8)
        )

        # --------------------------------------------------
        # 5. Vectorized softmax per source node
        #
        # Each source node has K outgoing edges.
        # Instead of looping over every node, calculate
        # the maximum and denominator for all nodes at once.
        # --------------------------------------------------

        num_nodes = node_features.shape[0]

        max_scores = torch.full(
            (num_nodes,),
            float("-inf"),
            device=node_features.device,
            dtype=attention_scores.dtype
        )

        max_scores.scatter_reduce_(
            0,
            source_nodes,
            attention_scores,
            reduce="amax",
            include_self=True
        )

        stabilized_scores = (
            attention_scores -
            max_scores[source_nodes]
        )

        exp_scores = torch.exp(
            stabilized_scores
        )

        sum_exp_scores = torch.zeros(
            num_nodes,
            device=node_features.device,
            dtype=attention_scores.dtype
        )

        sum_exp_scores.index_add_(
            0,
            source_nodes,
            exp_scores
        )

        attention = (
            exp_scores /
            (sum_exp_scores[source_nodes] + 1e-8)
        )

        attention = self.dropout(attention)

        # --------------------------------------------------
        # 6. Message passing
        # --------------------------------------------------

        messages = (
            target_features *
            attention.unsqueeze(-1)
        )

        output_features = torch.zeros(
            num_nodes,
            self.output_dim,
            device=node_features.device,
            dtype=projected_features.dtype
        )

        output_features.index_add_(
            0,
            source_nodes,
            messages
        )

        return output_features


class GAT(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=512,
        output_dim=512,
        dropout=0.1
    ):
        super().__init__()

        self.gat_layer_1 = GraphAttentionLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        self.gat_layer_2 = GraphAttentionLayer(
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features,
        edge_index,
        edge_weights
    ):
        # First GAT layer
        x = self.gat_layer_1(
            node_features,
            edge_index,
            edge_weights
        )

        x = F.relu(x)
        x = self.dropout(x)

        # Second GAT layer
        x = self.gat_layer_2(
            x,
            edge_index,
            edge_weights
        )

        return x
#construct spatial graph (rn m using only euclidean + rainfall + vv + slope - dont have info of extra stuff like river nearness , and stuff )
import torch
import torch.nn as nn


class DynamicGraphConstructor(nn.Module):

    def __init__(
            self,
            k=8
    ):

        super().__init__()

        self.k = k

    def forward(
            self,
            coordinates,
            environmental_features
    ):

        # coordinates:
        # [N, 2]
        #
        # environmental_features:
        # [N, 3]
        #
        # columns:
        # VV, rainfall, slope

        num_nodes = coordinates.shape[0]

        if num_nodes <= self.k:
            raise ValueError(
                "Number of nodes must be greater than k."
            )

        # 1. Euclidean distance between all anchors

        distances = torch.cdist(
            coordinates,
            coordinates,
            p=2
        )

        # Do not allow an anchor to select itself
        distances.fill_diagonal_(
            float("inf")
        )

        # 2. Find K nearest spatial neighbors

        neighbor_distances, neighbors = torch.topk(
            distances,
            k=self.k,
            dim=1,
            largest=False
        )

        # 3. Get environmental features of neighbors

        source_features = (
            environmental_features
            .unsqueeze(1)
            .expand(
                -1,
                self.k,
                -1
            )
        )

        neighbor_features = (
            environmental_features[neighbors]
        )

        # 4. Calculate feature differences

        feature_difference = torch.abs(
            source_features -
            neighbor_features
        )

        # 5. Normalize each environmental feature

        feature_std = (
            environmental_features.std(
                dim=0,
                keepdim=True
            ) + 1e-6
        )

        normalized_difference = (
            feature_difference /
            feature_std
        )

        # Average difference across:
        # VV, rainfall, slope

        feature_distance = (
            normalized_difference.mean(
                dim=2
            )
        )

        # 6. Normalize spatial distance

        distance_scale = (
            neighbor_distances.mean(
                dim=1,
                keepdim=True
            ) + 1e-6
        )

        normalized_distance = (
            neighbor_distances /
            distance_scale
        )

        
        # 7. Combine spatial + environmental similarity
       

        combined_distance = (
            normalized_distance +
            feature_distance
        ) / 2.0

        # Convert distance into similarity
        edge_weights = torch.exp(
            -combined_distance
        )

        # 8. Construct edge_index
       
        source_nodes = torch.arange(
            num_nodes,
            device=coordinates.device
        ).unsqueeze(1).expand(
            -1,
            self.k
        )

        edge_index = torch.stack(
            [
                source_nodes.reshape(-1),
                neighbors.reshape(-1)
            ],
            dim=0
        )

        edge_weights = edge_weights.reshape(-1)

        return edge_index, edge_weights
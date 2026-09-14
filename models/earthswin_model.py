import torch
import torch.nn as nn

from patch_pyramid import PatchPyramid
from gat import GAT
from temporal import TemporalTransformer


class EarthSwin(nn.Module):

    def __init__(
        self,
        in_channels=6,
        feature_dim=512,
        num_heads=8,
        temporal_layers=2,
        temporal_feedforward_dim=1024,
        dropout=0.1
    ):
        super().__init__()

        # Shared Patch Pyramid
        #
        # The same encoder processes all three weeks.
        self.patch_pyramid = PatchPyramid(
            in_channels=in_channels,
            scale_feature_dim=feature_dim
        )

        # Shared GAT
        #
        # The same GAT processes all three weeks.
        self.gat = GAT(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
            dropout=dropout
        )

        # Temporal Transformer
        #
        # Takes the three weekly spatial features.
        self.temporal_transformer = TemporalTransformer(
            input_dim=feature_dim,
            num_heads=num_heads,
            num_layers=temporal_layers,
            feedforward_dim=temporal_feedforward_dim,
            dropout=dropout,
            sequence_length=3
        )

        # Prediction head
        #
        # Produces one landslide logit for each spatial anchor.
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def process_week(
        self,
        patch_64,
        patch_128,
        patch_256,
        edge_index,
        edge_weights
    ):
        # Multi-scale feature extraction
        pyramid_features = self.patch_pyramid(
            patch_64,
            patch_128,
            patch_256
        )

        # Spatial graph reasoning
        spatial_features = self.gat(
            pyramid_features,
            edge_index,
            edge_weights
        )

        return spatial_features

    def forward(
        self,
        week_1,
        week_2,
        week_3,
        edge_index_1,
        edge_weights_1,
        edge_index_2,
        edge_weights_2,
        edge_index_3,
        edge_weights_3
    ):


        # Week 1
     

        week_1_features = self.process_week(
            week_1["patch_64"],
            week_1["patch_128"],
            week_1["patch_256"],
            edge_index_1,
            edge_weights_1
        )

        # Week 2

        week_2_features = self.process_week(
            week_2["patch_64"],
            week_2["patch_128"],
            week_2["patch_256"],
            edge_index_2,
            edge_weights_2
        )

        # Week 3

        week_3_features = self.process_week(
            week_3["patch_64"],
            week_3["patch_128"],
            week_3["patch_256"],
            edge_index_3,
            edge_weights_3
        )

        
        # Temporal modelling
        # 

        # [N, 3, 512]
        temporal_features = torch.stack(
            [
                week_1_features,
                week_2_features,
                week_3_features
            ],
            dim=1
        )

        # [N, 512]
        temporal_output = self.temporal_transformer(
            temporal_features
        )

        # Prediction
      

        # [N, 1]
        logits = self.classifier(
            temporal_output
        )

        # [N, 1]
        probabilities = torch.sigmoid(
            logits
        )

        return logits, probabilities
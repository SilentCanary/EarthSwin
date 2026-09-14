import torch
import torch.nn as nn


class ScaleFusion(nn.Module):

    def __init__(
            self,
            feature_dim=512,
            num_scales=3
    ):

        super().__init__()

        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, pyramid_features):

        # pyramid_features:
        # [B, 3, 512]

        scale_scores = self.scale_attention(
            pyramid_features
        )

        # [B, 3, 1]

        scale_weights = torch.softmax(
            scale_scores,
            dim=1
        )

        # Weighted combination of the 3 scales
        fused_features = (
            pyramid_features *
            scale_weights
        ).sum(dim=1)

        # [B, 512]

        return fused_features
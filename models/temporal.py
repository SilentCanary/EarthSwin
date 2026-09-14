import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim=512,
        num_heads=8,
        num_layers=2,
        feedforward_dim=1024,
        dropout=0.1,
        sequence_length=3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.sequence_length = sequence_length

        # Learnable temporal position embeddings
        self.temporal_position = nn.Parameter(
            torch.randn(1, sequence_length, input_dim)
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final normalization
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, temporal_features):
        # temporal_features: [B, 3, 512]

        if temporal_features.dim() != 3:
            raise ValueError(
                "Expected temporal features with shape [B, 3, 512]."
            )

        if temporal_features.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"got {temporal_features.shape[1]}."
            )

        if temporal_features.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, "
                f"got {temporal_features.shape[2]}."
            )

        # Add temporal position information
        x = temporal_features + self.temporal_position

        # Temporal self-attention
        x = self.transformer(x)

        # Normalize
        x = self.norm(x)

        # Use the final target week representation
        output = x[:, -1, :]

        return output
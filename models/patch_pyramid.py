import torch
import torch.nn as nn

from cnn import CNNFeatureExtractor
from swin import SwinFeatureExtractor
from attention import CrossModalAttention
from fusion import ScaleFusion

class PatchPyramid(nn.Module):

    def __init__(self,in_channels=6,cnn_dim=256,swin_dim=768,scale_feature_dim=512,output_dim=512):

        super().__init__()

        self.cnn = CNNFeatureExtractor(
            in_channels=in_channels
        )

        self.swin_64 = SwinFeatureExtractor(
            in_channels=in_channels,
            img_size=64
        )

        self.swin_128 = SwinFeatureExtractor(
            in_channels=in_channels,
            img_size=128
        )

        self.swin_256 = SwinFeatureExtractor(
            in_channels=in_channels,
            img_size=256
        )

        self.attention = CrossModalAttention(cnn_dim=cnn_dim,swin_dim=swin_dim,attention_dim=scale_feature_dim,num_heads=8)
        self.scale_fusion = ScaleFusion(feature_dim=scale_feature_dim,num_scales=3)
    def forward(self,patch_64,patch_128,patch_256):
        
        # 64 × 64
        cnn_64_features = self.cnn(patch_64)
        swin_64_features = self.swin_64(patch_64)

        fused_64 = self.attention(
            cnn_64_features,
            swin_64_features
        )

        # 128 × 128
        cnn_128_features = self.cnn(patch_128)
        swin_128_features = self.swin_128(patch_128)

        fused_128 = self.attention(
            cnn_128_features,
            swin_128_features
        )

        # 256 × 256
        cnn_256_features = self.cnn(patch_256)
        swin_256_features = self.swin_256(patch_256)

        fused_256 = self.attention(
            cnn_256_features,
            swin_256_features
        )

        # Combine all three scales
        pyramid_features = torch.stack(
            [
                fused_64,
                fused_128,
                fused_256
            ],
            dim=1
        )
        fused_features = self.scale_fusion(pyramid_features)
        return fused_features
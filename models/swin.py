import os
import sys
import torch
import torch.nn as nn

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
SWIN_MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "earthswin",
    "Swin-Transformer",
    "models",
)

sys.path.append(SWIN_MODELS_DIR)

try:
    from swin_transformer import SwinTransformer
except ModuleNotFoundError:
    import timm

class SwinFeatureExtractor(nn.Module):

    def __init__(self,in_channels=18,img_size=256):

        super().__init__()

        if "timm" in globals():
            self.swin = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                in_chans=in_channels,
                img_size=img_size,
                num_classes=0,
                global_pool="avg",
            )
        else:
            self.swin = SwinTransformer(img_size=img_size, patch_size=4, in_chans=in_channels,num_classes=0,embed_dim=96,depths=[2,2,6,2],num_heads = [3,6,12,24], window_size=8)

        self.feature_dim = self.swin.num_features

    def forward(self,x):
        if "timm" in globals():
            features = self.swin(x)
        else:
            features = self.swin.forward_features(x)

        return features
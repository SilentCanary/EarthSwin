import sys 
import torch 
import torch.nn as nn

SWIN_MODELS_DIR = ( "C:/Users/advit/Documents/major project/" "earthswin/Swin-Transformer/models" ) 

sys.path.append(SWIN_MODELS_DIR)


from swin_transformer import SwinTransformer

class SwinFeatureExtractor(nn.Module):

    def __init__(self,in_channels=18,img_size=256):

        super().__init__()

        self.swin = SwinTransformer(img_size=img_size, patch_size=4, in_chans=in_channels,num_classes=0,embed_dim=96,depths=[2,2,6,2],num_heads = [3,6,12,24], window_size=8)

        self.feature_dim = self.swin.num_features

    def forward(self,x):
        features = self.swin.forward_features(x)

        return features
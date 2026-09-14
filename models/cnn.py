import torch 
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):

    def __init__(self, in_channels=18):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x,1)

        return x
    
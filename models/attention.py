import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):

    def __init__(
            self,cnn_dim=256,swin_dim=768,attention_dim=512,num_heads=8
    ):
        super().__init__()

        self.cnn_projection = nn.Linear(cnn_dim,attention_dim)

        self.swin_projection = nn.Linear(swin_dim, attention_dim)

        self.attention = nn.MultiheadAttention(embed_dim=attention_dim,num_heads=num_heads,batch_first=True)

        self.norm = nn.LayerNorm(attention_dim)

    def forward(self,cnn_features,swin_features):

        cnn_features = self.cnn_projection(cnn_features)

        swin_features = self. swin_projection(swin_features)

        cnn_features = cnn_features.unsqueeze(1)
        swin_features = swin_features.unsqueeze(1)

        attended_features , _ = self.attention(query=cnn_features,key=swin_features,value=swin_features)

        # Residual connection
        attended_features = (
            attended_features +
            cnn_features
        )

        # Normalize
        attended_features = self.norm(
            attended_features
        )

        # Remove sequence dimension
        # [B,1,512] → [B,512]
        attended_features = attended_features.squeeze(1)

        return attended_features
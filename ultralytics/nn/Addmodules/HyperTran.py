import torch
import torch.nn as nn
import math
from ultralytics.nn.modules import Conv, C2f, Detect


class HyperTransformer(nn.Module):
    """??Transformer????"""
    def __init__(self, in_dim, hyper_dim=64, num_heads=8):
        super().__init__()
        # ??????

        self.hyper_edge = nn.Sequential(
            nn.Conv2d(in_dim, hyper_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_dim, hyper_dim, 1)
        )

        # Transformer??

        self.transformer = nn.TransformerEncoderLayer(
            d_model=hyper_dim,
            nhead=num_heads,
            dim_feedforward=hyper_dim*4,
            activation='gelu'
        )

        # ????

        self.fusion = nn.Conv2d(hyper_dim*2, in_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # ??????

        hyper_feat = self.hyper_edge(x)  # [B, K, H, W]


        # Transformer?? (HW??????)

        trans_feat = hyper_feat.flatten(2).permute(2, 0, 1)  # [HW, B, K]

        trans_feat = self.transformer(trans_feat)
        trans_feat = trans_feat.permute(1, 2, 0).view(B, -1, H, W)

        # ?????

        fused = torch.cat([hyper_feat, trans_feat], dim=1)
        return x + self.fusion(fused)

class HT_C2f(C2f):
    """???C2f??"""
    def __init__(self, c1, c2, k=1, hyper_dim=64, num_heads=8, **kwargs):
        super().__init__(c1, c2, **kwargs)
        # ?HyperTransformer????Bottleneck

        self.m = nn.ModuleList(
            HyperTransformer(c2//2, hyper_dim, num_heads) if i%2==0
            else self.m[i]
            for i in range(len(self.m))
        )

class HyperDetect(Detect):
    """??????"""
    def __init__(self, nc=80, ch=(256, 512, 1024), hyper_dims=(64, 128, 256)):
        super().__init__(nc, ch)
        self.trans_blocks = nn.ModuleList([
            HyperTransformer(c, hd)
            for c, hd in zip(ch, hyper_dims)
        ])

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.trans_blocks[i](x[i])
        return super().forward(x)



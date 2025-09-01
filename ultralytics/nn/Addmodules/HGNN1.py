import torch
import torch.nn as nn

from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data
import numpy as np

num_nodes = 100
num_features = 16

# 2. ?????? (????)
features = torch.randn(num_nodes, num_features)
# ?????? CSV ?????
edge_index_np = np.loadtxt("/home/ubuntu/v8/dadadazeng (3rd copy)/hyperedge_train.csv", delimiter=",", skiprows=1, dtype=np.int64)
edge_index = torch.tensor(edge_index_np.T, dtype=torch.long)
# ?? PyG ????
data = Data(x=features, edge_index=edge_index)  # ?? index ? long?????


class HGNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(HGNNBlock, self).__init__()
        self.hyper_conv = HypergraphConv(in_channels, out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ??????
        self.cached_edge_index = None
        self.cached_edge_weight = None
        self.cached_num_nodes = None

    def forward(self, feats, batch=None):
        """
        feats: [N, F] ?? patch ??? backbone ??
        batch: ????????? batch ?????????????
        """
        device = feats.device
        N = feats.size(0)  # ????

        # ?? cached_edge_index ???????????????????
        if self.cached_edge_index is None or self.cached_num_nodes != N:
            # ?????????????????????????????????
            # idx ? [E, 2]????????????
            idx = torch.stack([
                torch.arange(0, N - 1, dtype=torch.long),
                torch.arange(1, N, dtype=torch.long)
            ], dim=1)  # E = N - 1

            # ??? [2, E] ??? edge_index??? CUDA
            edge_index = idx.t().contiguous().to(device)

            # ????????1??? CUDA ?
            edge_weight = torch.ones(edge_index.size(1), device=device)

            # ??
            self.cached_edge_index = edge_index
            self.cached_edge_weight = edge_weight
            self.cached_num_nodes = N

        # ????
        hg_out = self.hyper_conv(feats, self.cached_edge_index, self.cached_edge_weight)

        # ?? & dropout
        out = self.activation(hg_out)
        out = self.dropout(out)
        return out
class C2fHGNN1(nn.Module):
    def __init__(self, c1, c2, edge_index_path=None):
        super().__init__()
        self.edge_index_path = edge_index_path
        self.cached_edge_index = None
        self.cached_edge_weight = None
        self.m = nn.ModuleList([
            HGNNBlock(c1, c2),
            HGNNBlock(c2, c2),
            HGNNBlock(c2, c2)
        ])
        self.edge_index_path = edge_index_path
        self.cached_edge_index = None
        self.cached_edge_weight = None

    def forward(self, x):
        y = [x]
        feats = x
        B, C, H, W = feats.shape
        feats = feats.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, HW, C]

        if self.cached_edge_index is None or self.cached_edge_weight is None:
            # ... ???? edge_index ???? ...
            pass

        # ?? edge_index ?????????
        device = feats.device
        if self.cached_edge_index is None and self.edge_index_path is not None:
            self.build_edge_index()  # ???????????
        # ?????? None ?????
        if self.cached_edge_index is not None:
            self.cached_edge_index = self.cached_edge_index.to(device)
        if self.cached_edge_weight is not None:
            self.cached_edge_weight = self.cached_edge_weight.to(device)

        for m in self.m:
            feats = m(feats, self.cached_edge_index, self.cached_edge_weight)

        feats = feats.permute(0, 2, 1).view(B, self.c, H, W)  # [B, C, H, W]
        y.append(feats)
        return torch.cat(y, 1)

class Conv(nn.Module):
    """????? (??YOLO????)"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,
                              autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """??YOLOv8 Bottleneck"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def autopad(k, p=None):  # ??YOLO?padding??
    return p if p is not None else k // 2
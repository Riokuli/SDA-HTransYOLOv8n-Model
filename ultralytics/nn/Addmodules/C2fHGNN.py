import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

# ---------------------- ???? ----------------------
def autopad(k, p=None, d=2):  # kernel, padding, dilation
    """???? 'same' padding"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    # ?? tuple ? Conv2d ??
    if isinstance(p, list):
        p = tuple(p)
    return p

class Conv(nn.Module):
    """?????Conv2d + BN + ??"""
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        )
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        """????: ???"""
        return self.conv(x)

class Bottleneck(nn.Module):
    """???????"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(1, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # 1x1 ??
        self.cv1 = Conv(c1, c_, k=k[0], s=1)
        # 3x3 ??
        self.cv2 = Conv(c_, c2, k=k[1], s=1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

# ---------------------- ???? ----------------------
def make_incidence_matrix(H: int, W: int, hyperedges: list) -> torch.Tensor:
    """
    ???????? (N x E)?N=H*W, E=len(hyperedges)
    hyperedges: ?????????????
    ?? dense tensor
    """
    N = H * W
    E = len(hyperedges)
    Hmat = torch.zeros(N, E)
    for e, nodes in enumerate(hyperedges):
        Hmat[nodes, e] = 1.0
    return Hmat

# ---------------------- C2f_HGNN ?? ----------------------
class C2f_HGNN(nn.Module):
    """
    C2f ??????????? HypergraphConv
    """

    def __init__(self,
                 c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = False,
                 g: int = 1,
                 e: float = 0.5,
                 hyperedge_feats: int = 128,
                 *,
                 incidence_matrix: torch.Tensor  # ???-only
                 ):
        """
        :param c1: ????
        :param c2: ????
        :param n: Bottleneck ????
        :param shortcut: ??????
        :param g: ???
        :param e: ???
        :param hyperedge_feats: ????????
        :param incidence_matrix: (N×E) ???????????????
        """
        super().__init__()
        # ????????? buffer
        self.register_buffer('incidence_matrix', incidence_matrix)

        # ????...
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, g, k=(1,3), e=1.0)
            for _ in range(n)
        ])
        in_hg, out_hg = (2 + n) * self.c, hyperedge_feats
        self.hg_conv = HypergraphConv(in_hg, out_hg)
        self.cv2 = Conv(out_hg, c2, k=1, s=1)

    def forward(self, x):
        # ?????
        x = self.cv1(x)
        x1, x2 = x.chunk(2, dim=1)
        y = [x1, x2]
        for b in self.m:
            y.append(b(y[-1]))

        z = torch.cat(y, dim=1)
        B, C, H, W = z.shape
        nodes = z.flatten(2).transpose(1, 2)
        # (B, N, C)
        incidence = torch.tensor(self.incidence_matrix, dtype=torch.long)         # (N, E)
        out = []
        for b in range(B):
            N = nodes.shape[1]
            edge_list = []
            he_id = 0
            for i in range(0, N, 3):
                for j in range(i, min(i + 3, N)):
                    edge_list.append([j, he_id])
                he_id += 1
            # ?? CUDA ????????
            incidence = torch.tensor(edge_list, dtype=torch.long, device=nodes.device).t().contiguous()
            out.append(self.hg_conv(nodes[b], incidence))
        hg_out = torch.stack(out, dim=0)            # (B, N, out_hg)
        hg_out = hg_out.transpose(1, 2).view(B, -1, H, W)
        return self.cv2(hg_out)

# ---------------------- ???? ----------------------
if __name__ == "__main__":
    # ?????? 64x64
    H, W = 64, 64
    # ??????????????
    hyperedges = [list(range(i*W, (i+1)*W)) for i in range(H)]
    incidence = make_incidence_matrix(H, W, hyperedges)
    # ????
    module = C2f_HGNN(c1=32, c2=64, incidence_matrix=incidence, n=2)
    x = torch.randn(1, 32, H, W)
    y = module(x)
    print(y.shape)  # (1, 64, 64, 64)

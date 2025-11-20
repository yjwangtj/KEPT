import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialBackboneLite(nn.Module):
    def __init__(self, in_channels=3, c_s=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, c_s, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_s),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.net(x)
        feat = F.interpolate(feat, (H, W), mode="bilinear", align_corners=False)
        return feat


class TemporalFusionEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_frames: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_frames + 1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, feats: torch.Tensor):
        B, T, C, H, W = feats.shape
        gap = feats.mean(dim=(-2, -1))
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, gap], dim=1)
        seq = seq + self.pos[:, : (T + 1)]
        out = self.encoder(seq)
        cls_tok = out[:, 0]
        return cls_tok, out

class SpatialOnlyEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 64,
        s_channels: int = 64,
        t_depth: int = 4,
        t_heads: int = 8,
        t_mlp_ratio: float = 4.0,
        t_dropout: float = 0.1,
        max_frames: int = 16,
    ):
        super().__init__()
        self.spatial = SpatialBackboneLite(in_channels=3, c_s=s_channels)
        self.align = nn.Sequential(
            nn.Conv2d(s_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal = TemporalFusionEncoder(
            dim=out_channels, depth=t_depth, heads=t_heads,
            mlp_ratio=t_mlp_ratio, dropout=t_dropout, max_frames=max_frames
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5 and x.size(1) >= 1,
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            xt = x[:, t]
            if xt.size(1) == 1:
                xt = xt.repeat(1, 3, 1, 1)
            s = self.spatial(xt)
            s = self.align(s)
            feats.append(s)
        feats = torch.stack(feats, dim=1)
        cls_tok, seq_out = self.temporal(feats)
        return cls_tok, seq_out

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H, W = 2, 7, 224, 224
    model = SpatialOnlyEncoder(out_channels=64, s_channels=64, max_frames=T)
    x_rgb = torch.randn(B, T, 3, H, W)
    cls, seq = model(x_rgb)
    print("Spatial-only:", cls.shape, seq.shape)

    x_gray = torch.randn(B, T, 1, H, W)
    cls2, seq2 = model(x_gray)
    print("Spatial-only (1ch):", cls2.shape, seq2.shape)
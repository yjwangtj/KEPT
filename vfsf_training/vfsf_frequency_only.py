import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Patchify2D(nn.Module):
    def __init__(self, patch_size: int = 32):
        super().__init__()
        self.ps = patch_size

    def unfold(self, x: torch.Tensor):
        B, C, H, W = x.shape
        pad_h = (self.ps - H % self.ps) % self.ps
        pad_w = (self.ps - W % self.ps) % self.ps
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        H2, W2 = x.shape[-2:]
        patches = F.unfold(x, kernel_size=self.ps, stride=self.ps)
        return patches, (H2, W2)

    def fold(self, patches: torch.Tensor, hw: Tuple[int, int], C: int) -> torch.Tensor:
        H2, W2 = hw
        out = F.fold(patches, output_size=(H2, W2), kernel_size=self.ps, stride=self.ps)
        return out

def to_gray(x: torch.Tensor) -> torch.Tensor:
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b

class FrequencyOnlyBranchFast(nn.Module):
    def __init__(self, patch_size=32, c_f=64, mlp_ratio=1.5):
        super().__init__()
        self.patch = Patchify2D(patch_size)
        self.ps = patch_size
        hidden = int(patch_size * patch_size * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(patch_size * patch_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, patch_size * patch_size),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(1, c_f, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_f)
        self.act = nn.ReLU(inplace=True)

        self.register_parameter("lf_gate", nn.Parameter(torch.tensor(0.33)))
        self.register_parameter("mf_gate", nn.Parameter(torch.tensor(0.34)))
        self.register_parameter("hf_gate", nn.Parameter(torch.tensor(0.33)))

        ps = patch_size
        yy, xx = torch.meshgrid(torch.arange(ps), torch.arange(ps), indexing='ij')
        center = (ps - 1) / 2.0
        radius = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
        r_max = max(radius.max().item(), 1.0)
        mask_l = (radius <= r_max * (1 / 3)).float()
        mask_m = ((radius > r_max * (1 / 3)) & (radius <= r_max * (2 / 3))).float()
        mask_h = (radius > r_max * (2 / 3)).float()
        self.register_buffer("mask_l", mask_l, persistent=False)
        self.register_buffer("mask_m", mask_m, persistent=False)
        self.register_buffer("mask_h", mask_h, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xg = to_gray(x)
        patches, hw = self.patch.unfold(xg)
        ps = self.ps
        L = patches.shape[-1]

        pr = patches.transpose(1, 2).reshape(B, L, 1, ps, ps)

        with torch.cuda.amp.autocast(enabled=False):
            pr32 = pr.float()
            x_c = torch.fft.fft2(pr32, dim=(-2, -1), norm="ortho")
            mag = x_c.abs()

            mask_l = self.mask_l.to(mag.dtype).to(mag.device)
            mask_m = self.mask_m.to(mag.dtype).to(mag.device)
            mask_h = self.mask_h.to(mag.dtype).to(mag.device)
            mag = (self.lf_gate.sigmoid() * mag * mask_l
                   + self.mf_gate.sigmoid() * mag * mask_m
                   + self.hf_gate.sigmoid() * mag * mask_h)

            mag_vec = mag.view(B * L, ps * ps)
            attn = self.mlp(mag_vec)
            mod = (mag_vec * attn).view(B, L, 1, ps, ps)

            w = mod.mean(dim=(-2, -1), keepdim=True)
            recon = pr32 * (1.0 + w)

            recon_vec = recon.view(B, L, ps * ps).transpose(1, 2).reshape(B, ps * ps, L)
            x_rec = self.patch.fold(recon_vec, hw, C=1)

        out = self.proj(x_rec)
        out = self.bn(out)
        out = self.act(out)
        return out

class TemporalFusionEncoder(nn.Module):
    def __init__(self, dim=64, depth=4, heads=8, mlp_ratio=4.0, dropout=0.1, max_frames=16):
        super().__init__()
        self.dim = dim
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_frames + 1, dim))
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, feats: torch.Tensor):
        B, T, C, H, W = feats.shape
        gap = feats.mean(dim=(-2, -1))
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, gap], dim=1)
        seq = seq + self.pos[:, : (T + 1)]
        out = self.encoder(seq)
        return out[:, 0], out

class FrequencyOnlyEncoder(nn.Module):
    def __init__(
        self,
        out_channels=64,
        f_patch_size=32,
        f_channels=64,
        f_mlp_ratio=1.5,
        t_depth=4, t_heads=8, t_mlp_ratio=4.0, t_dropout=0.1,
        max_frames=16,
    ):
        super().__init__()
        self.freq = FrequencyOnlyBranchFast(
            patch_size=f_patch_size, c_f=f_channels, mlp_ratio=f_mlp_ratio
        )
        self.align = nn.Sequential(
            nn.Conv2d(f_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal = TemporalFusionEncoder(
            dim=out_channels, depth=t_depth, heads=t_heads,
            mlp_ratio=t_mlp_ratio, dropout=t_dropout, max_frames=max_frames
        )

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            xt = x[:, t]
            f = self.freq(xt)
            f = self.align(f)
            feats.append(f)
        feats = torch.stack(feats, dim=1)
        return self.temporal(feats)

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H, W = 2, 3, 224, 224
    m = FrequencyOnlyEncoder(out_channels=64, f_patch_size=32, f_channels=64, max_frames=T)
    x = torch.randn(B, T, 3, H, W)
    cls, seq = m(x)
    print(cls.shape, seq.shape)
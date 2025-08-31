# fsf_temporal_7frame_fixed.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm      # pip install timm
from PIL import Image
import torchvision.transforms as transforms

# -------------------------
# 频域注意力 MLP
# -------------------------
class FreqAttentionMLP(nn.Module):
    def __init__(self, patch_size, hidden_dim=512):
        super().__init__()
        input_dim = patch_size * patch_size + 1
        output_dim = patch_size * patch_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B*L, P²+1)
        return torch.sigmoid(self.mlp(x))


# -------------------------
# 频域特征提取模块（已修正）
# -------------------------
class FrequencyDomainModule(nn.Module):
    """
    1) RGB→灰度；2) 非重叠 patch unfold；3) patch FFT→幅度谱；
    4) 对每个 patch 计算 GAP 并拼接谱 → MLP → patch attention；
    5) 加权后 fold 回原图尺寸。
    """
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.att_mlp = FreqAttentionMLP(patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 灰度 & pad 到 patch 整数倍
        x_gray = (0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]).unsqueeze(1)  # (B,1,H,W)
        pad_H = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_W = (self.patch_size - W % self.patch_size) % self.patch_size
        x_pad = F.pad(x_gray, (0, pad_W, 0, pad_H), mode='reflect')
        _, _, H_p, W_p = x_pad.shape

        # 2. Unfold → patches: (B, P², L)
        patches = F.unfold(x_pad, kernel_size=self.patch_size, stride=self.patch_size)
        L = patches.shape[-1]  # number of patches

        # 3. 变成 (B, L, 1, P, P) 做 FFT → (B, L, 1, P, P)
        patches_2d = patches.view(B, 1, self.patch_size, self.patch_size, L).permute(0,4,1,2,3)
        fft = torch.fft.fft2(patches_2d, dim=(-2, -1))
        mag = torch.abs(fft).view(B, L, -1)  # (B, L, P²)

        # 4. 计算每 patch 的 GAP → (B, L, 1)
        gap = mag.mean(dim=2, keepdim=True)

        # 5. 拼接 mag & gap → (B, L, P²+1) → (B*L, P²+1) for MLP
        att_in = torch.cat([mag, gap], dim=2).view(B * L, -1)  # (B*L, P²+1)
        att_out = self.att_mlp(att_in)                         # (B*L, P²)
        att_w = att_out.view(B, L, 1, self.patch_size, self.patch_size)  # (B, L,1,P,P)

        # 6. 加权 & fold 回图
        weighted = torch.abs(fft) * att_w                       # (B, L,1,P,P)
        weighted = weighted.permute(0,2,3,4,1).reshape(B, self.patch_size*self.patch_size, L)
        freq_feat = F.fold(weighted,
                           output_size=(H_p, W_p),
                           kernel_size=self.patch_size,
                           stride=self.patch_size)               # (B,1,H_p,W_p)

        # 7. 裁剪回原尺寸
        return freq_feat[:, :, :H, :W]                         # (B,1,H,W)


# -------------------------
# 空域特征提取模块
# -------------------------
class SpatialDomainModule(nn.Module):
    def __init__(self, local_checkpoint=None):
        super().__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
        if local_checkpoint and os.path.exists(local_checkpoint):
            if local_checkpoint.endswith('.safetensors'):
                from safetensors.torch import load_file as safe_load
                sd = safe_load(local_checkpoint)
            else:
                sd = torch.load(local_checkpoint, map_location='cpu')
            self.swin.load_state_dict(sd, strict=False)
        self.swin.reset_classifier(0)

    def forward(self, x):
        feat = self.swin.forward_features(x)                    # (B, C_sp, H/4, W/4)
        feat_up = F.interpolate(feat, size=x.shape[-2:],
                                mode='bilinear', align_corners=False)
        return feat_up                                          # (B, C_sp, H, W)


# -------------------------
# 频-空 融合主干
# -------------------------
class FreqSpatialFusion(nn.Module):
    def __init__(self, out_channels=64, patch_size=16, local_checkpoint=None):
        super().__init__()
        self.freq_module = FrequencyDomainModule(patch_size)
        self.spa_module = SpatialDomainModule(local_checkpoint)

        # infer spatial channels
        with torch.no_grad():
            dummy = torch.zeros(1,3,224,224)
            C_sp = self.spa_module(dummy).shape[1]

        self.fusion_conv = nn.Conv2d(1 + C_sp, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        f = self.freq_module(x)       # (B,1,H,W)
        s = self.spa_module(x)        # (B,C_sp,H,W)
        z = torch.cat([f, s], dim=1)  # (B,1+C_sp,H,W)
        y = self.fusion_conv(z)
        return self.act(self.bn(y))   # (B,out_channels,H,W)


# -------------------------
# 时序 Transformer Fusion
# -------------------------
class TemporalFusionEncoder(nn.Module):
    def __init__(self,
                 fsf_encoder: nn.Module,
                 dim: int,
                 depth: int = 4,
                 heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 max_frames: int = 7):
        super().__init__()
        self.fsf = fsf_encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames+1, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # 帧内特征
        x_in = x.view(B * T, C, H, W)
        feat = self.fsf(x_in)                  # (B*T, dim, H, W)
        feat = self.pool(feat).view(B, T, -1)  # (B, T, dim)

        # CLS token + pos emb
        cls = self.cls_token.expand(B, -1, -1)       # (B,1,dim)
        tokens = torch.cat([cls, feat], dim=1)       # (B, T+1, dim)
        tokens = tokens + self.pos_emb[:, :T+1, :]   # add position

        # Transformer expects (seq_len, batch, dim)
        out = self.temporal_encoder(tokens.permute(1, 0, 2))
        return out[0]                                # (B, dim)


# -------------------------
# 辅助：加载并预处理图像
# -------------------------
def load_image(path, size=(224,224)):
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(path).convert('RGB')
    return tf(img)


# -------------------------
# main 示例：读取当前文件夹下 7 张图片，并编码
# -------------------------
if __name__ == '__main__':
    # 1. 列出并取前 7 张图片
    exts = ('.jpg','.jpeg','.png','.bmp')
    files = sorted(f for f in os.listdir('.') if f.lower().endswith(exts))[:7]

    # 2. 加载并堆成 (1,7,3,224,224)
    imgs = [load_image(f) for f in files]
    seq = torch.stack(imgs, dim=0).unsqueeze(0)

    # 3. 初始化模型
    fsf = FreqSpatialFusion(out_channels=64, patch_size=16, local_checkpoint=None)
    temp_enc = TemporalFusionEncoder(
        fsf_encoder=fsf,
        dim=64,
        depth=4,
        heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        max_frames=7
    )

    # 4. 推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp_enc.to(device)
    seq = seq.to(device)

    temp_enc.eval()
    with torch.no_grad():
        video_feat = temp_enc(seq)  # (1, 64)

    print("读取的文件：", files)
    print("输出特征维度：", video_feat.shape)

    vec = video_feat.squeeze(0).cpu().numpy()
    print("Feature vector values:\n", vec)
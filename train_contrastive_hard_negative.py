# train_contrastive_hard_negative.py

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.optim import AdamW

from vfsf import FreqSpatialFusion, TemporalFusionEncoder

# ---------------------------------
# 1. 自监督对比数据集
# ---------------------------------
class ContrastiveJSONDataset(Dataset):
    def __init__(self, json_path, transform):
        """
        加载原 JSON，只用 'images' 字段，忽略 'trajectories'。
        对每条序列做两次随机增强，得到正例对。
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # 1) 取前 7 帧
        paths = entry['images'][:7]
        # 2) 随机增强两次
        views = []
        for _ in range(2):
            imgs = []
            for p in paths:
                img = Image.open(p).convert('RGB')
                imgs.append(self.transform(img))
            views.append(torch.stack(imgs, dim=0))  # (7,3,H,W)
        # 返回 (view1, view2)
        return views[0], views[1]


# ---------------------------------
# 2. 投影头（Projection Head）
# ---------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)  # L2 归一化后用于对比


# ---------------------------------
# 3. InfoNCE Loss + Hard Negative Mining
# ---------------------------------
def contrastive_loss_hard(emb_anchor, emb_pos, temperature=0.07, top_k=5):
    """
    emb_anchor, emb_pos: (B, D) 已归一化
    1) 正例相似度 sim_pos = cos(anchor, pos) / temperature
    2) 负例：batch 内所有其他 emb_pos
    3) 计算所有 neg_sim = cos(anchor, neg) / temp，选 top_k 最大（最难负样本）
    4) loss = -log( exp(sim_pos) / (exp(sim_pos) + sum_{neg in hard} exp(neg_sim)) )
    """
    B, D = emb_anchor.shape
    # 相似度矩阵 (B, B)
    sim_matrix = emb_anchor @ emb_pos.T    # since both L2 normed
    sim_matrix = sim_matrix / temperature

    # 正例在对角线上
    sim_pos = torch.diag(sim_matrix)       # (B,)

    # 对每个 i，取 neg_sim 行 i，排除自己，然后 top_k
    losses = []
    for i in range(B):
        negs = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])  # (B-1,)
        hard_negs, _ = torch.topk(negs, k=min(top_k, B-1))
        denom = sim_pos[i].exp() + hard_negs.exp().sum()
        loss_i = - (sim_pos[i].exp() / denom).log()
        losses.append(loss_i)
    return torch.stack(losses).mean()


# ---------------------------------
# 4. 训练脚本
# ---------------------------------
def main():
    # 配置
    json_path   = '3_ready4embedding_train.json'
    batch_size  = 8
    val_split   = 0.1
    epochs      = 50
    lr_encoder  = 1e-5
    lr_proj     = 1e-4
    weight_decay= 1e-4
    temperature = 0.07
    top_k       = 10
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据增强：空间 + 颜色
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # 数据集
    full_ds = ContrastiveJSONDataset(json_path, transform)
    n_val = int(len(full_ds) * val_split)
    n_trn = len(full_ds) - n_val
    trn_ds, val_ds = random_split(full_ds, [n_trn, n_val])

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型 + 投影头
    fsf = FreqSpatialFusion(out_channels=64, patch_size=16, local_checkpoint=None)
    encoder = TemporalFusionEncoder(fsf_encoder=fsf,
                                    dim=64, depth=4, heads=8,
                                    mlp_ratio=4.0, dropout=0.1,
                                    max_frames=7).to(device)
    proj_head = ProjectionHead(in_dim=64, hidden_dim=256, out_dim=128, dropout=0.3).to(device)

    # 优化器
    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': lr_encoder},
        {'params': proj_head.parameters(),'lr': lr_proj},
    ], weight_decay=weight_decay)

    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        encoder.train(); proj_head.train()
        total_loss = 0.0
        for v1, v2 in trn_loader:
            v1, v2 = v1.to(device), v2.to(device)   # (B,7,3,224,224) ×2
            optimizer.zero_grad()
            # 提取特征
            f1 = encoder(v1)  # (B,64)
            f2 = encoder(v2)  # (B,64)
            z1 = proj_head(f1)  # (B,128)
            z2 = proj_head(f2)  # (B,128)
            # 计算 InfoNCE + Hard Neg Mining
            loss = contrastive_loss_hard(z1, z2, temperature, top_k)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * v1.size(0)

        train_loss = total_loss / len(trn_loader.dataset)

        # 验证（同训练，但不更新）
        encoder.eval(); proj_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v1, v2 in val_loader:
                v1, v2 = v1.to(device), v2.to(device)
                z1 = proj_head(encoder(v1))
                z2 = proj_head(encoder(v2))
                val_loss += contrastive_loss_hard(z1, z2, temperature, top_k).item() * v1.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder':    encoder.state_dict(),
                'proj_head':  proj_head.state_dict(),
                'optimizer':  optimizer.state_dict(),
            }, 'best_contrastive_hard.pth')
            print("  ↳ Saved best model.")

    print("Training finished. Best Val Loss:", best_val_loss)


if __name__ == '__main__':
    main()
# train_contrastive_memory_hard.py

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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from vfsf import FreqSpatialFusion, TemporalFusionEncoder

# ---------------------------------
# 1. 对比学习 Dataset（读取原 JSON，忽略轨迹）
# ---------------------------------
class ContrastiveJSONDataset(Dataset):
    def __init__(self, json_path, transform):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        frames = entry['images'][:7]
        # 两路随机增强
        views = []
        for _ in range(2):
            imgs = []
            for p in frames:
                img = Image.open(p).convert('RGB')
                imgs.append(self.transform(img))
            views.append(torch.stack(imgs, dim=0))  # (7,3,H,W)
        return views[0], views[1]


# ---------------------------------
# 2. Projection Head
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
        return F.normalize(self.net(x), dim=1)


# ---------------------------------
# 3. Main training with memory bank + hard negative mining
# ---------------------------------
def main():
    # -------- Config --------
    json_path    = '3_ready4embedding_train.json'
    batch_size   = 8
    val_split    = 0.1
    epochs       = 50
    lr_encoder   = 1e-5
    lr_proj      = 1e-4
    weight_decay = 1e-4
    temperature  = 0.07
    hard_top_k   = 10
    queue_size   = 1024
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------- Transforms --------
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # -------- Data --------
    full_ds = ContrastiveJSONDataset(json_path, transform)
    n_val = int(len(full_ds) * val_split)
    n_trn = len(full_ds) - n_val
    trn_ds, val_ds = random_split(full_ds, [n_trn, n_val])
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # -------- Model & Projection --------
    fsf = FreqSpatialFusion(out_channels=64, patch_size=16, local_checkpoint=None)
    encoder = TemporalFusionEncoder(
        fsf_encoder=fsf,
        dim=64, depth=4, heads=8,
        mlp_ratio=4.0, dropout=0.1,
        max_frames=7
    ).to(device)

    proj_head = ProjectionHead(in_dim=64, hidden_dim=256, out_dim=128, dropout=0.3).to(device)

    # -------- Optimizer --------
    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': lr_encoder},
        {'params': proj_head.parameters(),'lr': lr_proj},
    ], weight_decay=weight_decay)

    # -------- Memory Bank --------
    memory_bank = torch.randn(queue_size, 128, device=device)
    memory_bank = F.normalize(memory_bank, dim=1)
    bank_ptr = 0

    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        encoder.train(); proj_head.train()
        total_loss = 0.0

        for v1, v2 in trn_loader:
            v1, v2 = v1.to(device), v2.to(device)

            optimizer.zero_grad()
            f1 = encoder(v1)               # (B,64)
            f2 = encoder(v2)               # (B,64)
            z1 = proj_head(f1)             # (B,128)
            z2 = proj_head(f2)             # (B,128)

            # positive similarity
            sim_pos = (z1 * z2).sum(dim=1, keepdim=True) / temperature  # (B,1)

            # in-batch negatives
            sim_ib = (z1 @ z2.T) / temperature                       # (B,B)
            negs_ib = []
            B = z1.size(0)
            for i in range(B):
                row = torch.cat([sim_ib[i,:i], sim_ib[i,i+1:]], dim=0)
                negs_ib.append(row)
            negs_ib = torch.stack(negs_ib, dim=0)                     # (B,B-1)

            # memory-bank negatives
            sim_mb = (z1 @ memory_bank.T) / temperature               # (B, Q)

            # combine and pick hard top_k
            hard_negs = []
            for i in range(B):
                all_negs = torch.cat([negs_ib[i], sim_mb[i]], dim=0)
                topk_vals, _ = torch.topk(all_negs, k=min(hard_top_k, all_negs.size(0)), largest=True)
                hard_negs.append(topk_vals)
            hard_negs = torch.stack(hard_negs, dim=0)                 # (B, top_k)

            # compute InfoNCE
            exp_pos = sim_pos.exp()                                   # (B,1)
            exp_negs = hard_negs.exp().sum(dim=1, keepdim=True)       # (B,1)
            loss = - (exp_pos / (exp_pos + exp_negs)).log().mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B

            # enqueue z2
            with torch.no_grad():
                bs = z2.size(0)
                if bank_ptr + bs <= queue_size:
                    memory_bank[bank_ptr:bank_ptr+bs] = z2
                    bank_ptr = (bank_ptr + bs) % queue_size
                else:
                    end = queue_size - bank_ptr
                    memory_bank[bank_ptr:] = z2[:end]
                    memory_bank[:bs-end] = z2[end:]
                    bank_ptr = bs - end

        train_loss = total_loss / len(trn_loader.dataset)

        # Validate
        encoder.eval(); proj_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v1, v2 in val_loader:
                v1, v2 = v1.to(device), v2.to(device)
                z1 = proj_head(encoder(v1))
                z2 = proj_head(encoder(v2))

                sim_pos = (z1 * z2).sum(dim=1, keepdim=True) / temperature
                sim_ib = (z1 @ z2.T) / temperature
                negs_ib = []
                B = z1.size(0)
                for i in range(B):
                    row = torch.cat([sim_ib[i,:i], sim_ib[i,i+1:]], dim=0)
                    negs_ib.append(row)
                negs_ib = torch.stack(negs_ib, dim=0)
                sim_mb = (z1 @ memory_bank.T) / temperature

                hard_negs = []
                for i in range(B):
                    all_negs = torch.cat([negs_ib[i], sim_mb[i]], dim=0)
                    topk_vals, _ = torch.topk(all_negs, k=min(hard_top_k, all_negs.size(0)), largest=True)
                    hard_negs.append(topk_vals)
                hard_negs = torch.stack(hard_negs, dim=0)

                exp_pos = sim_pos.exp()
                exp_negs = hard_negs.exp().sum(dim=1, keepdim=True)
                val_loss += - (exp_pos / (exp_pos + exp_negs)).log().mean().item() * B

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder':    encoder.state_dict(),
                'proj_head':  proj_head.state_dict(),
                'optimizer':  optimizer.state_dict(),
            }, 'best_contrastive_memory_hard.pth')
            print("  ↳ Saved best model.")

    print("Training complete. Best Val Loss:", best_val_loss)


if __name__ == '__main__':
    main()
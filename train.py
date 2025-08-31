# train_regression_optimized_fixed.py

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR

from vfsf import FreqSpatialFusion, TemporalFusionEncoder

# ----------------------------
# 1. 深度回归头
# ----------------------------
class RegressionHead(nn.Module):
    """
    MLP 回归头：两层隐藏 + LayerNorm + ReLU + Dropout → 输出 out_dim
    """
    def __init__(self, in_dim=64, hidden_dims=(256, 128), out_dim=12, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------
# 2. Dataset：加载 JSON 并规范化目标
# ----------------------------
class TrajectoryJSONDataset(Dataset):
    def __init__(self, data, transform, mean, std):
        """
        data:     list of dicts loaded from JSON
        transform: torchvision transforms for images
        mean, std: tensors of shape (12,) for target normalization
        """
        self.data = data
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # 1) 加载 7 帧图像
        imgs = []
        for img_path in entry['images']:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        seq = torch.stack(imgs, dim=0)  # (7,3,H,W)

        # 2) 解析 trajectories，只取 x,y，按时间升序
        traj_dict = json.loads(entry['trajectories'])
        ordered = sorted(traj_dict.items(),
                         key=lambda x: float(x[0].split()[0]))
        vec = []
        for _, v in ordered:
            vec.extend([v['x'], v['y']])
        target = torch.tensor(vec, dtype=torch.float)    # (12,)

        # 3) 归一化
        target_norm = (target - self.mean) / self.std

        return seq, target_norm


# ----------------------------
# 3. 计算目标均值与标准差
# ----------------------------
def compute_target_stats(json_path):
    with open(json_path, 'r') as f:
        raw = json.load(f)
    all_vec = []
    for entry in raw:
        traj = json.loads(entry['trajectories'])
        ordered = sorted(traj.items(), key=lambda x: float(x[0].split()[0]))
        vec = []
        for _, v in ordered:
            vec.extend([v['x'], v['y']])
        all_vec.append(vec)
    all_targets = torch.tensor(all_vec, dtype=torch.float)  # (N,12)
    mean = all_targets.mean(dim=0)
    std  = all_targets.std(dim=0)
    return raw, mean, std


# ----------------------------
# 4. 主函数：数据、模型、训练流程
# ----------------------------
def main():
    # --- Config ---
    json_path     = '3_ready4embedding_train.json'
    batch_size    = 8
    val_split     = 0.2
    num_workers   = 4
    encoder_lr    = 1e-5
    head_lr       = 1e-4
    weight_decay  = 1e-4
    num_epochs    = 30
    freeze_epochs = 5  # 前几 epoch 冻结编码器，只训练回归头
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Data & Stats ---
    raw_data, mean, std = compute_target_stats(json_path)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    dataset = TrajectoryJSONDataset(raw_data, transform, mean, std)
    train_size = int((1 - val_split) * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # --- Model & Head ---
    fsf = FreqSpatialFusion(out_channels=64, patch_size=16, local_checkpoint=None)
    model = TemporalFusionEncoder(
        fsf_encoder=fsf,
        dim=64, depth=4, heads=8,
        mlp_ratio=4.0, dropout=0.1,
        max_frames=7
    ).to(device)

    reg_head = RegressionHead(
        in_dim=64, hidden_dims=(256,128),
        out_dim=6*2, dropout=0.3
    ).to(device)

    # 冻结编码器参数，只训练回归头
    for p in model.parameters():
        p.requires_grad = False

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(),      'lr': encoder_lr},
        {'params': reg_head.parameters(),   'lr': head_lr},
    ], weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss(reduction='mean')

    best_val_rmse = float('inf')

    # --- Training Loop ---
    for epoch in range(1, num_epochs+1):
        # 解冻编码器用于微调
        if epoch == freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            print(f"Epoch {epoch}: Unfroze encoder for fine-tuning.")

        # Train
        model.train(); reg_head.train()
        train_loss = 0.0
        for seqs, targets_norm in train_loader:
            seqs, targets_norm = seqs.to(device), targets_norm.to(device)
            optimizer.zero_grad()
            feats = model(seqs)               # (B,64)
            preds_norm = reg_head(feats)      # (B,12)

            loss = criterion(preds_norm, targets_norm)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seqs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validate & compute real RMSE
        model.eval(); reg_head.eval()
        val_loss = 0.0
        sq_err_real = 0.0
        count_real = 0

        with torch.no_grad():
            for seqs, targets_norm in val_loader:
                seqs, targets_norm = seqs.to(device), targets_norm.to(device)
                feats = model(seqs)
                preds_norm = reg_head(feats)

                val_loss += criterion(preds_norm, targets_norm).item() * seqs.size(0)

                # 反归一化
                preds_real = preds_norm * std.to(device) + mean.to(device)
                targets_real = targets_norm * std.to(device) + mean.to(device)
                sq_err_real += ((preds_real - targets_real)**2).sum().item()
                count_real += preds_real.numel()

        val_loss /= len(val_loader.dataset)
        val_rmse_real = (sq_err_real / count_real)**0.5

        print(f"Epoch {epoch:02d} | "
              f"Train MSE(norm): {train_loss:.4f} | "
              f"Val MSE(norm):   {val_loss:.4f} | "
              f"Val RMSE:        {val_rmse_real:.4f} m")

        # Save best
        if val_rmse_real < best_val_rmse:
            best_val_rmse = val_rmse_real
            torch.save({
                'encoder_state': model.fsf.state_dict(),
                'temporal_state': model.temporal_encoder.state_dict(),
                'reg_head':      reg_head.state_dict(),
                'optimizer':     optimizer.state_dict(),
            }, 'best_regression_optimized_fixed.pth')
            print("  ↳ Saved best model weights.")

        scheduler.step()

    print(f"Training complete. Best Val RMSE: {best_val_rmse:.4f} m")


if __name__ == '__main__':
    main()
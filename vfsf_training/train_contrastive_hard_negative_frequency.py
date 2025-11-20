import json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.optim import AdamW

from vfsf_frequency_only import FrequencyOnlyEncoder

class ContrastiveJSONDataset(Dataset):
    def __init__(self, json_path, transform, max_frames=7):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        self.max_frames = max_frames

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]['images'][: self.max_frames]
        views = []
        for _ in range(2):
            imgs = []
            for p in paths:
                img = Image.open(p).convert('RGB')
                imgs.append(self.transform(img))
            views.append(torch.stack(imgs, dim=0))
        return views[0], views[1]

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return F.normalize(self.net(x), dim=1)

def contrastive_loss_hard(emb_anchor, emb_pos, temperature=0.07, top_k=10):
    B, D = emb_anchor.shape
    sim = (emb_anchor @ emb_pos.T) / temperature
    sim_pos = torch.diag(sim)
    losses = []
    for i in range(B):
        negs = torch.cat([sim[i, :i], sim[i, i+1:]])
        hard_negs, _ = torch.topk(negs, k=min(top_k, B-1))
        denom = sim_pos[i].exp() + hard_negs.exp().sum()
        losses.append(-(sim_pos[i].exp() / denom).log())
    return torch.stack(losses).mean()

def main():
    json_path   = '/home/wyj/0-Research_Projects/RATP/vfsf_training/3_ready4embedding_train.json'
    batch_size  = 8
    val_split   = 0.1
    epochs      = 50
    lr_encoder  = 1e-5
    lr_proj     = 1e-4
    weight_decay= 1e-4
    temperature = 0.07
    top_k       = 10
    T           = 7
    img_size    = 224
    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp     = torch.cuda.is_available()

    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    full_ds = ContrastiveJSONDataset(json_path, transform, max_frames=T)
    n_val = max(1, int(len(full_ds) * val_split))
    n_trn = len(full_ds) - n_val
    trn_ds, val_ds = random_split(full_ds, [n_trn, n_val])

    common_args = dict(num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,  **common_args)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common_args)

    encoder = FrequencyOnlyEncoder(
        out_channels=64,
        f_patch_size=32,
        f_channels=64,
        f_mlp_ratio=1.5,
        t_depth=4, t_heads=8, t_mlp_ratio=4.0, t_dropout=0.1,
        max_frames=T
    ).to(device)
    proj_head = ProjectionHead(in_dim=64, hidden_dim=256, out_dim=128, dropout=0.3).to(device)

    optimizer = AdamW([
        {'params': encoder.parameters(),   'lr': lr_encoder},
        {'params': proj_head.parameters(), 'lr': lr_proj},
    ], weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        encoder.train(); proj_head.train()
        total_loss = 0.0

        for v1, v2 in trn_loader:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                f1, _ = encoder(v1)
                f2, _ = encoder(v2)
                z1 = proj_head(f1)
                z2 = proj_head(f2)
                loss = contrastive_loss_hard(z1, z2, temperature=temperature, top_k=top_k)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * v1.size(0)

        train_loss = total_loss / len(trn_loader.dataset)

        encoder.eval(); proj_head.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for v1, v2 in val_loader:
                v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
                f1, _ = encoder(v1)
                f2, _ = encoder(v2)
                z1 = proj_head(f1); z2 = proj_head(f2)
                val_loss += contrastive_loss_hard(z1, z2, temperature=temperature, top_k=top_k).item() * v1.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'encoder':   encoder.state_dict(),
                'proj_head': proj_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {
                    'T': T,
                    'out_channels': 64,
                    'proj_out_dim': 128,
                    'temperature': temperature,
                    'top_k': top_k,
                    'f_patch_size': 32,
                    'f_mlp_ratio': 1.5
                }
            }, 'best_contrastive_hard_frequency_only.pth')
            print("  ↳ Saved best model.")

    print("Done. Best Val:", best_val)

if __name__ == '__main__':
    main()
# encode_sequences_to_pkl.py

import os
import json
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# import your encoder definitions
from vfsf import FreqSpatialFusion, TemporalFusionEncoder

# ----------------------------
# ProjectionHead definition (must match training)
# ----------------------------
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
        return nn.functional.normalize(self.net(x), dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load trained weights ---
    ckpt = torch.load('best_contrastive_hard.pth', map_location=device)
    fsf = FreqSpatialFusion(out_channels=64, patch_size=16, local_checkpoint=None)
    encoder = TemporalFusionEncoder(
        fsf_encoder=fsf,
        dim=64, depth=4, heads=8,
        mlp_ratio=4.0, dropout=0.1,
        max_frames=7
    ).to(device)
    proj_head = ProjectionHead(in_dim=64, hidden_dim=256, out_dim=128, dropout=0.3).to(device)

    encoder.load_state_dict(ckpt['encoder'])
    proj_head.load_state_dict(ckpt['proj_head'])
    encoder.eval()
    proj_head.eval()

    # --- Prepare transform (deterministic) ---
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    # --- Read the new JSON ---
    with open('3.5_refined_train0824.json', 'r') as f:
        entries = json.load(f)

    embeddings = []
    # optional: keep a list of identifiers
    ids = []

    with torch.no_grad():
        for idx, entry in enumerate(entries):
            img_paths = entry['images'][:7]
            # load and preprocess
            imgs = [transform(Image.open(p).convert('RGB')) for p in img_paths]
            seq = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # (1,7,3,224,224)
            # forward
            feat = encoder(seq)            # (1,64)
            emb = proj_head(feat)          # (1,128)
            emb_np = emb.cpu().squeeze(0).numpy()  # (128,)
            embeddings.append(emb_np)
            ids.append(idx)

    # --- Save to pickle ---
    out = {
        'ids': ids,               # list of entry indices
        'embeddings': embeddings  # list of numpy arrays
    }
    with open('database_embeddings_refined0824.pkl', 'wb') as pf:
        pickle.dump(out, pf)

    print(f"Saved {len(embeddings)} embeddings to database_embeddings_refined.pkl")

if __name__ == '__main__':
    main()
import torch
from vfsf import load_image, FreqSpatialFusion, TemporalFusionEncoder

# 1. 准备你的 7 张图像路径（按时间顺序排列）
image_paths = [
    "frames/frame01.jpg",
    "frames/frame02.jpg",
    "frames/frame03.jpg",
    "frames/frame04.jpg",
    "frames/frame05.jpg",
    "frames/frame06.jpg",
    "frames/frame07.jpg",
]

# 2. 加载并预处理：resize→to_tensor→normalize
imgs = [load_image(p, size=(224,224)) for p in image_paths]
# imgs 是长度 7 的 list，每个元素 shape=(3,224,224)

# 3. 拼 batch 维度和帧维度
#    torch.stack 之后 shape=(7,3,224,224)，再 unsqueeze 出 batch 维 → (1,7,3,224,224)
seq = torch.stack(imgs, dim=0).unsqueeze(0)

# 4. 初始化模型（out_channels、dim 必须对应）
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

# 如果有 GPU，就把数据和模型都搬过去
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
temp_enc.to(device)
seq = seq.to(device)

# 5. 前向计算
temp_enc.eval()
with torch.no_grad():
    video_feat = temp_enc(seq)  # 输出 shape = (1, 64)

print("Encoded feature shape:", video_feat.shape)

vec = video_feat.squeeze(0).cpu().numpy()
print("Feature vector values:\n", vec)
# 如果你想要一个一维向量，可以取 video_feat[0] → torch.Size([64])
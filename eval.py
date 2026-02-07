import torch
from data_loader.dtu import DTUDataset
from models.encoder import Encoder
from models.fusion import MultiScaleFusion
from models.gan import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DTUDataset("data/DTU", num_views=3)

encoder = Encoder().to(device)
fusion = MultiScaleFusion().to(device)
generator = Generator().to(device)

encoder.eval()
fusion.eval()
generator.eval()

with torch.no_grad():
    views = dataset[0].unsqueeze(0).to(device)
    feats = torch.stack([encoder(views[:, v]) for v in range(views.shape[1])], dim=1)
    fused = fusion(feats)
    output = generator(fused)

print("Evaluation successful. Output shape:", output.shape)

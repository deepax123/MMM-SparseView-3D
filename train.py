import yaml
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from data_loader.dtu import DTUDataset
from models.encoder import Encoder
from models.fusion import MultiScaleFusion
from models.gan import Generator

# Load config
with open("configs/dtu_3view.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DTUDataset(
    cfg["dataset"]["root_dir"],
    cfg["dataset"]["num_views"]
)
loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"])

encoder = Encoder().to(device)
fusion = MultiScaleFusion().to(device)
generator = Generator().to(device)

optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(fusion.parameters()) +
    list(generator.parameters()),
    lr=cfg["training"]["lr"]
)

loss_fn = torch.nn.L1Loss()

for epoch in range(cfg["training"]["epochs"]):
    for views in loader:
        views = views.to(device)
        B, V, C, H, W = views.shape

        feats = []
        for v in range(V):
            feats.append(encoder(views[:, v]))

        feats = torch.stack(feats, dim=1)
        fused = fusion(feats)
        output = generator(fused)

        loss = loss_fn(output, views[:, 0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

print("Training complete.")

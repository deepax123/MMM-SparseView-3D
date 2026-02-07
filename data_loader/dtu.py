import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DTUDataset(Dataset):
    def __init__(self, root, views=3):
        self.root = root
        self.views = views
        self.scans = sorted(os.listdir(root))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_path = os.path.join(self.root, self.scans[idx], "images")
        images = sorted(os.listdir(scan_path))[:self.views]

        imgs = []
        for img in images:
            img = Image.open(os.path.join(scan_path, img)).convert("RGB")
            imgs.append(self.transform(img))

        imgs = torch.stack(imgs)  # [V, 3, H, W]

        gt_depth = torch.zeros(1, 256, 256)  # placeholder GT
        return imgs, gt_depth

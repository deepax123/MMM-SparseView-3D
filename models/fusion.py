import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self, views):
        super().__init__()
        self.fuse = nn.Conv2d(64 * views, 64, 1)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, feats):
        fused = torch.cat(feats, dim=1)
        return self.out(self.fuse(fused))

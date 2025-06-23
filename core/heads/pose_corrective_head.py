import torch 
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange



class PoseCorrectiveHead(nn.Module):
    def __init__(self, condition_dim, in_channels=3, out_channels=3):
        super().__init__()
        self.condition_proj = nn.Linear(condition_dim, 512)
        
        # U-Net encoder
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # U-Net decoder with condition injection
        self.dec3 = nn.Conv2d(256 + 512, 128, 3, padding=1)  # +512 for condition
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, out_channels, 3, padding=1)
        
    def forward(self, x, condition):
        # Project condition to spatial features
        cond_feat = self.condition_proj(condition)  # (B, 512)
        cond_feat = cond_feat.view(cond_feat.shape[0], -1, 1, 1)  # (B, 512, 1, 1)
        cond_feat = cond_feat.expand(-1, -1, x.shape[2], x.shape[3])  # (B, 512, H, W)
        
        # Encoder
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2)))
        e3 = F.relu(self.enc3(F.max_pool2d(e2, 2)))
        
        # Decoder with condition
        d3 = F.relu(self.dec3(torch.cat([e3, cond_feat], dim=1)))
        d2 = F.relu(self.dec2(F.interpolate(d3, scale_factor=2)))
        d1 = self.dec1(F.interpolate(d2, scale_factor=2))
        
        return d1

import torch 
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

import sys 
sys.path.append('.')
from core.backbone.resnet import resnet18
from hmr.hmr_head import HMRHead


class HMR(nn.Module):
    def __init__(self, smpl_model):
        """
        Given a batch of normal images, predict pixel-aligned canonical space position and uv coordinates
        """
        super(HMR, self).__init__()
        self.smpl_model = smpl_model

        self.backbone = resnet18(in_channels=3, pretrained=True)
        self.head = HMRHead()



    def forward(self, images):

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        images = rearrange(images, 'b n c h w -> (b n) c h w')

        features = self.backbone(images)

        features = rearrange(features, '(b n) c -> b n c', b=B, n=N)

        pred = self.head(features)

        return pred 



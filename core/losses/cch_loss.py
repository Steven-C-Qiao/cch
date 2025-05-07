import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

class CCHLoss(nn.Module):
    def __init__(self, single_directional=True):
        super().__init__()
        self.single_directional = single_directional

    def forward(self, v, v_pred):
        return chamfer_distance(v_pred, v, single_directional=self.single_directional)

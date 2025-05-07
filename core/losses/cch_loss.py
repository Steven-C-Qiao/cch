import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

class CCHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v, v_pred):
        return chamfer_distance(v_pred, v, single_directional=True)

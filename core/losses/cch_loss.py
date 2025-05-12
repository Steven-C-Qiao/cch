import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

from einops import rearrange


class PointMapLoss(nn.Module):
    def __init__(self, single_directional=True):
        super().__init__()
        self.single_directional = single_directional

    def forward(self, v, v_pred, mask=None):
        pass

class CCHLoss(nn.Module):
    def __init__(self, single_directional=True):
        super().__init__()
        self.single_directional = single_directional

    def forward(self, v, v_pred, mask=None, pred_dw=None):
        loss, loss_normals = chamfer_distance(v_pred, v, 
                                single_directional=self.single_directional, 
                                batch_reduction=None, 
                                point_reduction=None)

        loss_v_to_v_pred = loss[0]
        loss_v_pred_to_v = loss[1]
        

        if mask is not None:
            masked_loss_v_pred_to_v = loss_v_pred_to_v * rearrange(mask, 'b n c h w -> (b n) (c h w)')
            masked_loss_v_pred_to_v = masked_loss_v_pred_to_v.mean()
        else:
            masked_loss_v_pred_to_v = loss_v_pred_to_v.mean()

        
        if pred_dw is not None:
            loss_w = torch.mean(pred_dw ** 2)
        
        
        loss = masked_loss_v_pred_to_v + loss_v_to_v_pred.mean() + loss_w

        return loss, loss_normals

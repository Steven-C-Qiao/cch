import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

from einops import rearrange

class CCHLoss(nn.Module):
    def __init__(self, single_directional=True):
        super().__init__()
        self.single_directional = single_directional

    def forward(self, v, v_pred, mask=None):
        loss, loss_normals = chamfer_distance(v_pred, v, 
                                single_directional=self.single_directional, 
                                batch_reduction=None, 
                                point_reduction=None)

        loss_v_to_v_pred = loss[0]
        loss_v_pred_to_v = loss[1]

        # ipdb> loss[0].shape
        # torch.Size([8, 6890])
        # ipdb> loss[1].shape
        # torch.Size([8, 50176])


        # print(loss_v_to_v_pred.shape)
        # print(loss_v_pred_to_v.shape)

        # import ipdb; ipdb.set_trace()
        

        if mask is not None:
            masked_loss_v_pred_to_v = loss_v_pred_to_v * rearrange(mask, 'b n c h w -> (b n) (c h w)')
            masked_loss_v_pred_to_v = masked_loss_v_pred_to_v.mean()
        else:
            masked_loss_v_pred_to_v = loss_v_pred_to_v.mean()

        loss = masked_loss_v_pred_to_v + loss_v_to_v_pred.mean()
        # import ipdb; ipdb.set_trace()
        return loss, loss_normals

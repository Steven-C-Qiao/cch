import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

from einops import rearrange


class CanonicalRGBLoss(nn.Module):
    """
    Masked L2 loss for canonical color maps
    """
    def __init__(self):
        super().__init__()

    def forward(self, vc, vc_pred, mask=None):
        loss = torch.mean((vc - vc_pred) ** 2)
        if mask is not None:
            loss = loss * rearrange(mask, 'b n c h w -> (b n) (c h w)')
        return loss.mean()

class PosedPointmapChamferLoss(nn.Module):
    """
    Masked chamfer loss for posed points
    """
    def __init__(self, cfg):
        super().__init__()
        self.single_directional = cfg.LOSS.CHAMFER_SINGLE_DIRECTIONAL

    def forward(self, v, v_pred, mask=None):
        loss, loss_normals = chamfer_distance(v_pred, v, 
                                single_directional=self.single_directional, 
                                batch_reduction=None, 
                                point_reduction=None)

        loss_v_pred_to_v = loss[0]
        loss_v_to_v_pred = loss[1]
        
        if mask is not None:
            masked_loss_v_pred_to_v = loss_v_pred_to_v * rearrange(mask, 'b n c h w -> (b n) (c h w)')
            masked_loss_v_pred_to_v = masked_loss_v_pred_to_v.mean()
        else:
            masked_loss_v_pred_to_v = loss_v_pred_to_v.mean()

        loss = masked_loss_v_pred_to_v + loss_v_to_v_pred.mean()

        return loss# , loss_normals
        


class CCHLoss(nn.Module):
    """
    vc_gt: (B, N, H, W, 3)
    vc_pred: (B, N, H, W, 3)
    mask: (B, N, H, W)
    vp: (B, N, 6890, 3)
    vp_pred: (B, N, H, W, 3)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.posed_pointmap_loss = PosedPointmapChamferLoss(cfg)
        self.canonical_rgb_loss = CanonicalRGBLoss()

    def forward(self, vp, vp_pred, vc, vc_pred, mask=None, pred_dw=None):
        loss_dict = {}

        posed_loss = self.posed_pointmap_loss(vp, vp_pred, mask) * self.cfg.LOSS.VP_LOSS_WEIGHT
        canonical_loss = self.canonical_rgb_loss(vc, vc_pred, mask) * self.cfg.LOSS.VC_LOSS_WEIGHT
        loss_dict['posed_loss'] = posed_loss
        loss_dict['canonical_loss'] = canonical_loss
        
        total_loss = posed_loss + canonical_loss
        
        if pred_dw is not None:
            loss_w = torch.mean(pred_dw ** 2)
            loss_dict['w_loss'] = loss_w
            total_loss += loss_w * self.cfg.LOSS.W_REGULARISER_WEIGHT

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
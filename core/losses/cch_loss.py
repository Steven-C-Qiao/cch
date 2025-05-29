import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

from einops import rearrange

class CanonicalRGBConfLoss(nn.Module):
    """
    Masked L2 loss for canonical color maps
    
    vc: (B, N, H, W, 3)
    vc_pred: (B, N, H, W, 3)
    conf: (B, N, H, W) in [0, 1]
    mask: (B, N, H, W)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, vc, vc_pred, conf=None, mask=None):

        conf, log_conf = self.get_conf_log(conf)
        
        loss = torch.norm(vc - vc_pred, dim=-1) 

        conf_loss = loss * conf - self.alpha * log_conf

        if mask is not None:
            conf_loss = conf_loss * mask.squeeze()


        return conf_loss.mean()

class PosedPointmapChamferLoss(nn.Module):
    """
    Masked chamfer loss for posed points

    v_gt: (B, V1, 3)
    v_pred: (B, V2, 3)
    mask: (B, V2)
    """
    def __init__(self, cfg):
        super().__init__()
        self.single_directional = cfg.LOSS.CHAMFER_SINGLE_DIRECTIONAL
    
    def forward(self, v_gt, v_pred, mask=None):
        loss, loss_normals = chamfer_distance(v_pred, v_gt, 
                                single_directional=self.single_directional, 
                                batch_reduction=None, 
                                point_reduction=None)

        loss_v_pred_to_v = loss[0]
        loss_v_to_v_pred = loss[1]
        
        if mask is not None:
            masked_loss_v_pred_to_v = loss_v_pred_to_v * mask
            masked_loss_v_pred_to_v = masked_loss_v_pred_to_v.mean()
        else:
            masked_loss_v_pred_to_v = loss_v_pred_to_v.mean()

        loss = masked_loss_v_pred_to_v + loss_v_to_v_pred.mean()

        return loss, masked_loss_v_pred_to_v
    
class RegulariserLoss(nn.Module):
    """
    Regulariser loss for skinning weights
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, dw_pred):
        loss1 = torch.mean(torch.abs(torch.sum(dw_pred, dim=-1)))
        loss2 = torch.mean(dw_pred ** 2)
        return loss1 + loss2
        


class CCHLoss(nn.Module):
    """
    vc_gt: (B, N, H, W, 3)
    vc_pred: (B, N, H, W, 3)
    conf: (B, N, H, W) in [0, 1]
    mask: (B, N, H, W)
    vp: (B, N, 6890, 3)
    vp_pred: (B, N, V, 3)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.posed_pointmap_loss = PosedPointmapChamferLoss(cfg)
        self.canonical_rgb_loss = CanonicalRGBConfLoss()
        self.regulariser_loss = RegulariserLoss()


    def forward(self, vp, vp_pred, vc, vc_pred, conf, mask=None, dw_pred=None):
        loss_dict = {}

        posed_loss, chamfer_vp_pred_to_vp = self.posed_pointmap_loss(rearrange(vp, 'b n v c -> (b n) v c'), 
                                                                     rearrange(vp_pred, 'b n v c -> (b n) v c'), 
                                                                     rearrange(mask, 'b n h w -> (b n) (h w)')) 
        posed_loss *= self.cfg.LOSS.VP_LOSS_WEIGHT
        loss_dict['posed_loss'] = posed_loss
        loss_dict['chamfer_vp_pred_to_vp'] = chamfer_vp_pred_to_vp

        canonical_loss = self.canonical_rgb_loss(vc, vc_pred, conf=conf, mask=mask)
        canonical_loss *= self.cfg.LOSS.VC_LOSS_WEIGHT
        loss_dict['canonical_loss'] = canonical_loss
        
        total_loss = posed_loss + canonical_loss
        
        if dw_pred is not None:
            loss_dw = self.regulariser_loss(dw_pred) * self.cfg.LOSS.W_REGULARISER_WEIGHT
            loss_dict['dw_reg_loss'] = loss_dw
            total_loss += loss_dw

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
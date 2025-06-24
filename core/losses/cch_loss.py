import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
import pytorch_lightning as pl
from einops import rearrange

class CanonicalRGBConfLoss(nn.Module):
    """
    Masked L2 loss for canonical color maps
    
    vc: (B, N, H, W, 3)
    vc_pred: (B, N, H, W, 3)
    conf: (B, N, H, W) in [0, 1]
    mask: (B, N, H, W)
    """
    def __init__(self, cfg):
        self.alpha = cfg.LOSS.ALPHA
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

        loss_v_pred_to_v = torch.clamp(loss_v_pred_to_v, max=100)
        loss_v_to_v_pred = torch.clamp(loss_v_to_v_pred, max=100)
        
        if mask is not None:
            masked_loss_v_pred_to_v = loss_v_pred_to_v * mask
            masked_loss_v_pred_to_v = masked_loss_v_pred_to_v.mean()
        else:
            masked_loss_v_pred_to_v = loss_v_pred_to_v.mean()

        loss = masked_loss_v_pred_to_v + loss_v_to_v_pred.mean()

        return loss
        

class SkinningWeightLoss(nn.Module):
    """
    Loss for skinning weights
    """
    def __init__(self):
        super().__init__()

    def forward(self, w_smpl, w_pred, mask=None):
        loss = torch.nn.functional.mse_loss(w_pred, w_smpl, reduction='none')

        # if mask is not None:
            # loss = loss * mask[..., None]
            
        return loss.mean()
        
        


class CCHLoss(pl.LightningModule):
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
        self.canonical_rgb_loss = CanonicalRGBConfLoss(cfg)
        self.skinning_weight_loss = SkinningWeightLoss()

        self._create_loss_weight_schedule(cfg)


    def forward(self, vp, vp_pred, vc, vc_pred, conf, mask=None, w_pred=None, w_smpl=None, dvc_pred=None, dvc_conf=None, epoch=None):
        loss_dict = {}

        posed_loss = self.posed_pointmap_loss(
            rearrange(vp, 'b n v c -> (b n) v c'), 
            rearrange(vp_pred, 'b n v c -> (b n) v c'), 
            rearrange(mask, 'b n h w -> (b n) (h w)')
        ) 
        posed_loss *= self.posed_loss_schedule[epoch]
        loss_dict['vp_chamfer_loss'] = posed_loss

        canonical_loss = self.canonical_rgb_loss(vc, vc_pred, conf=conf, mask=mask)
        canonical_loss *= self.canonical_loss_schedule[epoch]
        loss_dict['vc_pm_loss'] = canonical_loss
        
        total_loss = posed_loss + canonical_loss
        
        if w_pred is not None:
            loss_w = self.skinning_weight_loss(w_smpl, w_pred, mask) * self.w_reg_loss_schedule[epoch]
            loss_dict['w_reg_loss'] = loss_w
            total_loss += loss_w

        if dvc_pred is not None:
            # l2 loss for dvc_pred
            dvc_loss = torch.norm(dvc_pred, dim=-1)
            dvc_loss = dvc_loss.mean() * self.dvc_loss_schedule[epoch]
            loss_dict['dvc_reg_loss'] = dvc_loss
            total_loss += dvc_loss

        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def _create_loss_weight_schedule(self, cfg):
        """
        Create a loss weight schedule for the loss weights
        """
        total_epochs = cfg.TRAIN.NUM_EPOCHS
        self.posed_loss_schedule = torch.linspace(cfg.LOSS.VP_LOSS_WEIGHT, cfg.LOSS.VP_LOSS_WEIGHT_FINAL, total_epochs)
        self.canonical_loss_schedule = torch.linspace(cfg.LOSS.VC_LOSS_WEIGHT, cfg.LOSS.VC_LOSS_WEIGHT_FINAL, total_epochs)
        self.w_reg_loss_schedule = torch.linspace(cfg.LOSS.W_REGULARISER_WEIGHT, cfg.LOSS.W_REGULARISER_WEIGHT_FINAL, total_epochs)
        self.dvc_loss_schedule = torch.linspace(cfg.LOSS.DVC_LOSS_WEIGHT, cfg.LOSS.DVC_LOSS_WEIGHT_FINAL, total_epochs)


if __name__ == '__main__':
    # test chamfer dist
    x = torch.tensor([[[0, 0, 0], [1, 1, 1]]]).float()
    y = torch.tensor([[[0.5, 0.5, 0.5]]]).float()
    loss, loss_normals = chamfer_distance(x, y, 
                                single_directional=False, 
                                batch_reduction=None, 
                                point_reduction=None)
    reduced_loss, reduced_loss_normals = chamfer_distance(x, y)
    print(loss)  # (tensor([[0.7500, 0.7500]]), tensor([[0.7500]]))
    print(reduced_loss)  # tensor(1.5000)
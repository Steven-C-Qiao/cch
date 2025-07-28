import torch
import torch.nn as nn
import pytorch_lightning as pl

from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds


class CCHLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.posed_chamfer_loss = MaskedChamferLoss()
        self.canonical_chamfer_loss = MaskedChamferLoss()

        self.vc_pm_loss = MaskedL2Loss()

        self.skinning_weight_loss = MaskedL2Loss()
        self.dvc_loss = MaskedL2Loss()



    def forward(self, predictions, batch):
        loss_dict = {}
        total_loss = 0

        B, N, H, W, _ = predictions['vc_init'].shape
        K = N 

        if "vc_init" in predictions:
            gt_vc = Pointclouds(
                points=batch['template_mesh_verts']
            )
            pred_vc = predictions['vc_init']
            mask = batch['masks']

            pred_vc = rearrange(pred_vc, 'b n h w c -> b (n h w) c')
            mask = rearrange(mask, 'b n h w -> b (n h w)')

            vc_loss = self.canonical_chamfer_loss(
                gt_vc, 
                pred_vc, 
                mask
            )
            vc_loss *= self.cfg.LOSS.VC_LOSS_WEIGHT
            loss_dict['vc_chamfer_loss'] = vc_loss
            total_loss = total_loss + vc_loss

        if "vc_init" in predictions:
            pred_vc = predictions['vc_init']
            gt_vc_smpl_pm = batch['vc_maps']

            mask = batch['smpl_mask']
            vc_pm_loss = self.vc_pm_loss(
                pred_vc,
                gt_vc_smpl_pm,
                mask
            )
            vc_pm_loss *= 10.
            loss_dict['vc_pm_loss'] = vc_pm_loss
            total_loss = total_loss + vc_pm_loss

        if "vp_init" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp_init']
            mask = batch['masks']

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_loss = self.posed_chamfer_loss(
                gt_vp,
                pred_vp, 
                mask
            ) 
            vp_loss *= self.cfg.LOSS.VP_LOSS_WEIGHT
            loss_dict['vp_init_chamfer_loss'] = vp_loss
            total_loss = total_loss + vp_loss

        if "vp" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp']
            mask = batch['masks']

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_loss = self.posed_chamfer_loss(
                gt_vp,
                pred_vp, 
                mask
            ) 
            vp_loss *= self.cfg.LOSS.VP_LOSS_WEIGHT
            loss_dict['vp_chamfer_loss'] = vp_loss
            total_loss = total_loss + vp_loss
  

        if "w" in predictions:
            pred_w = predictions['w']
            gt_w = batch['smpl_w_maps']
            mask = batch['masks']

            w_loss = self.skinning_weight_loss(
                gt_w, 
                pred_w, 
                mask
            )
            w_loss *= self.cfg.LOSS.W_REGULARISER_WEIGHT
            loss_dict['w_loss'] = w_loss
            total_loss = total_loss + w_loss

        loss_dict['total_loss'] = total_loss

        # for k, v in loss_dict.items():
        #     print(k, v.item())
        
        return total_loss, loss_dict
    

class MaskedChamferLoss(nn.Module):
    """
    Masked chamfer loss

    Args:
        x_gt: (B, V1, 3)
        x_pred: (B, V2, 3)
        mask: (B, V2, 1)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_gt, x_pred, mask):
        assert x_pred.shape[:2] == mask.shape[:2]
        
        loss, _ = chamfer_distance(
            x_pred, x_gt, 
            batch_reduction=None, 
            point_reduction=None
        )

        loss_pred2gt = loss[0]
        loss_gt2pred = loss[1]

        masked_loss_pred2gt = loss_pred2gt * mask

        return masked_loss_pred2gt.mean() + loss_gt2pred.mean()
            

class MaskedL2Loss(nn.Module):
    """
    Masked L2 loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y, mask=None):
        loss = torch.norm(x - y, dim=-1)

        if mask is not None:
            loss = loss * mask
        return loss.sum() / mask.sum()


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

# class PosedPointmapChamferLoss(nn.Module):
#     """
#     Masked chamfer loss for posed points

#     v_gt: (B, V1, 3)
#     v_pred: (B, V2, 3)
#     mask: (B, V2)
#     """
#     def __init__(self, cfg):
#         super().__init__()
#         self.single_directional = cfg.LOSS.CHAMFER_SINGLE_DIRECTIONAL
    
#     def forward(self, v_gt, v_pred, mask=None):
#         loss, loss_normals = chamfer_distance(
#             v_pred, v_gt, 
#             single_directional=self.single_directional, 
#             batch_reduction=None, 
#             point_reduction=None
#         )

#         loss_vpp2vp = loss[0]
#         loss_vp2vpp = loss[1]

#         loss_vpp2vp = torch.clamp(loss_vpp2vp, max=100)
#         loss_vp2vpp = torch.clamp(loss_vp2vpp, max=100)
        
#         if mask is not None:
#             masked_loss_vpp2vp = loss_vpp2vp * mask
#             masked_loss_vpp2vp = masked_loss_vpp2vp.mean()
#         else:
#             masked_loss_vpp2vp = loss_vpp2vp.mean()

#         loss = masked_loss_vpp2vp + loss_vp2vpp.mean()

#         return loss
    

        

# class SkinningWeightLoss(nn.Module):
#     """
#     Loss for skinning weights
#     """
#     def __init__(self):
#         super().__init__()

#     def forward(self, w_smpl, w_pred, mask=None):
#         loss = torch.nn.functional.mse_loss(w_pred, w_smpl, reduction='none')

#         # if mask is not None:
#             # loss = loss * mask[..., None]
            
#         return loss.mean()
        
        



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
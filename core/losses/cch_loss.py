import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds


class CCHLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.posed_chamfer_loss = MaskedUncertaintyChamferLoss()
        self.canonical_chamfer_loss = MaskedUncertaintyChamferLoss()

        self.vc_pm_loss = MaskedUncertaintyL2Loss()

        self.skinning_weight_loss = MaskedUncertaintyL2Loss()
        self.dvc_loss = MaskedUncertaintyL2Loss()



    def forward(self, predictions, batch):
        loss_dict = {}
        total_loss = 0

        B, N, H, W, _ = predictions['vc_init'].shape
        K = 5 



        # if "vc_init" in predictions:
        #     gt_vc = Pointclouds(
        #         points=batch['template_mesh_verts']
        #     )
        #     pred_vc = predictions['vc_init']
        #     mask = batch['masks'][:, :N]
        #     confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None

        #     pred_vc = rearrange(pred_vc, 'b n h w c -> b (n h w) c')
        #     mask = rearrange(mask, 'b n h w -> b (n h w)')
        #     if confidence is not None:
        #         confidence = rearrange(confidence, 'b n h w -> b (n h w)')

        #     vc_loss = self.canonical_chamfer_loss(
        #         gt_vc, 
        #         pred_vc, 
        #         mask,
        #         confidence
        #     )
        #     vc_loss *= self.cfg.LOSS.VC_CHAMFER_LOSS_WEIGHT
        #     loss_dict['vc_chamfer_loss'] = vc_loss
        #     total_loss = total_loss + vc_loss


        if "vc_init" in predictions:
            pred_vc = predictions['vc_init']
            gt_vc_smpl_pm = batch['vc_maps'][:, :N]
            mask = batch['smpl_mask'][:, :N]
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None
            
            vc_pm_loss = self.vc_pm_loss(
                pred_vc,
                gt_vc_smpl_pm,
                mask,
                confidence
            )
            vc_pm_loss *= self.cfg.LOSS.VC_PM_LOSS_WEIGHT
            loss_dict['vc_pm_loss'] = vc_pm_loss
            total_loss = total_loss + vc_pm_loss


        if "w" in predictions:
            pred_w = predictions['w']
            gt_w = batch['smpl_w_maps'][:, :N]
            mask = batch['masks'][:, :N]
            confidence = predictions['w_conf'] if "w_conf" in predictions else None

            w_loss = self.skinning_weight_loss(
                gt_w, 
                pred_w, 
                mask,
                # confidence
            )
            w_loss *= self.cfg.LOSS.W_REGULARISER_WEIGHT
            loss_dict['w_loss'] = w_loss
            total_loss = total_loss + w_loss

        if "vp_init" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp_init']
            mask = batch['masks']
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None

            # mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            mask = rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            if confidence is not None:
                confidence = rearrange(confidence[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')

            vp_loss = self.posed_chamfer_loss(
                gt_vp,
                pred_vp, 
                mask,
                # confidence
            ) 
            vp_loss *= self.cfg.LOSS.VP_INIT_CHAMFER_LOSS_WEIGHT
            loss_dict['vp_init_chamfer_loss'] = vp_loss
            total_loss = total_loss + vp_loss

        if "vp" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp']
            mask = batch['masks']
            confidence = predictions['dvc_conf'] if "dvc_conf" in predictions else None

            # mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            mask = rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            if confidence is not None:
                # confidence = rearrange(confidence[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
                confidence = rearrange(confidence, 'b k n h w -> (b k) (n h w)')

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')

            vp_loss = self.posed_chamfer_loss(
                gt_vp,
                pred_vp, 
                mask,
                # confidence
            ) 
            vp_loss *= self.cfg.LOSS.VP_CHAMFER_LOSS_WEIGHT
            loss_dict['vp_chamfer_loss'] = vp_loss
            total_loss = total_loss + vp_loss
  

        loss_dict['total_loss'] = total_loss

        # for k, v in loss_dict.items():
        #     print(k, v.item())
        
        return total_loss, loss_dict
    

class MaskedUncertaintyChamferLoss(nn.Module):
    """
    Masked chamfer loss

    Args:
        x_gt: (B, V1, 3)
        x_pred: (B, V2, 3)
        mask: (B, V2, 1)
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, x_gt, x_pred, mask, confidence=None):
        assert x_pred.shape[:2] == mask.shape[:2]

        mask_flat = mask.squeeze(-1).bool()  # (B, V2)
        
        x_pred_list = [x_pred[b][mask_flat[b]] for b in range(x_pred.shape[0])]
        conf_list = [confidence[b][mask_flat[b]] for b in range(x_pred.shape[0])] if confidence is not None else None

        x_pred_ptclds = Pointclouds(points=x_pred_list)


        loss, _ = chamfer_distance(
            x_pred_ptclds, x_gt, 
            batch_reduction=None, 
            point_reduction=None
        )


        loss_pred2gt = loss[0]
        loss_gt2pred = loss[1]

        
        if confidence is not None:
            conf_padded = torch.zeros_like(loss_pred2gt)
            log_conf_padded = torch.zeros_like(loss_pred2gt)
            
            for b in range(len(conf_list)):
                num_valid_points = len(conf_list[b])
                if num_valid_points > 0:
                    conf, log_conf = self.get_conf_log(conf_list[b])
                    conf_padded[b, :num_valid_points] = conf
                    log_conf_padded[b, :num_valid_points] = log_conf


            loss_pred2gt = loss_pred2gt * conf_padded - self.alpha * log_conf_padded

        # masked_loss_pred2gt = loss_pred2gt * mask
        # return masked_loss_pred2gt.mean() + loss_gt2pred.mean()
        return loss_pred2gt.mean() + loss_gt2pred.mean()
    


class MaskedUncertaintyL2Loss(nn.Module):
    """
    Masked L2 loss weighted by uncertainty 
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, x, y, mask=None, uncertainty=None):

        loss = torch.norm(x - y, dim=-1)

        if uncertainty is not None:
            conf, log_conf = self.get_conf_log(uncertainty)
            loss = loss * conf - self.alpha * log_conf

        if mask is not None:
            loss = loss * mask

        return loss.sum() / mask.sum()
    


class MaskedUncertaintyExpL2Loss(nn.Module):
    def __init__(self, alpha=1.0):
        """
        Loss for finetuning Vc_pm, penalise only the large deviations from SMPL 
        """
        super().__init__()
        self.alpha = alpha

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, x, y, mask=None, uncertainty=None):
        offset = 3

        loss = torch.exp(torch.norm(x - y, dim=-1) - offset) - np.exp( - offset)

        if uncertainty is not None:
            conf, log_conf = self.get_conf_log(uncertainty)
            loss = loss * conf - self.alpha * log_conf

        if mask is not None:
            loss = loss * mask

        return loss.sum() / mask.sum()
    



# class CanonicalRGBConfLoss(nn.Module):
#     """
#     Masked L2 loss for canonical color maps
    
#     vc: (B, N, H, W, 3)
#     vc_pred: (B, N, H, W, 3)
#     conf: (B, N, H, W) in [0, 1]
#     mask: (B, N, H, W)
#     """
#     def __init__(self, cfg):
#         self.alpha = cfg.LOSS.ALPHA
#         super().__init__()

#     def get_conf_log(self, x):
#         return x, torch.log(x)

#     def forward(self, vc, vc_pred, conf=None, mask=None):

#         conf, log_conf = self.get_conf_log(conf)
        
#         loss = torch.norm(vc - vc_pred, dim=-1) 

#         conf_loss = loss * conf - self.alpha * log_conf

#         if mask is not None:
#             conf_loss = conf_loss * mask.squeeze()


#         return conf_loss.mean()

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
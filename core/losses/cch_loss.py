import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds

from core.utils.general import check_and_fix_inf_nan
from core.utils.loss_utils import filter_by_quantile


class CCHLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.posed_chamfer_loss = MaskedUncertaintyChamferLoss()
        self.canonical_chamfer_loss = MaskedUncertaintyChamferLoss()

        self.vc_pm_l2_loss = MaskedUncertaintyL2Loss()
        self.vc_pm_asap_loss = ASAPLoss()

        self.skinning_weight_loss = MaskedUncertaintyL2Loss()
        self.dvc_loss = MaskedUncertaintyL2Loss()



    def forward(self, predictions, batch, dataset_name=None):
        loss_dict = {}
        total_loss = 0

        B, N, H, W, _ = predictions['vc_init'].shape
        K = 5 

        if "vc_init" in predictions and "template_mesh_verts" in batch:
            gt_vc = Pointclouds(
                points=batch['template_mesh_verts']
            )
            pred_vc = predictions['vc_init']
            mask = batch['masks'][:, :N]
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None

            pred_vc = rearrange(pred_vc, 'b n h w c -> b (n h w) c')
            mask = rearrange(mask, 'b n h w -> b (n h w)')
            if confidence is not None:
                confidence = rearrange(confidence, 'b n h w -> b (n h w)')

            vc_loss = self.canonical_chamfer_loss(
                gt_vc, 
                pred_vc, 
                mask,
                confidence
            )
            vc_loss *= self.cfg.LOSS.VC_CHAMFER_LOSS_WEIGHT
            loss_dict['vc_chamfer_loss'] = vc_loss
            total_loss = total_loss + vc_loss


        if "vc_init" in predictions and "vc_maps" in batch:
            assert dataset_name == '4DDress'
            loss_fn = self.vc_pm_l2_loss
            # loss_fn = self.vc_pm_asap_loss

                
            pred_vc = predictions['vc_init']
            gt_vc_smpl_pm = batch['vc_maps'][:, :N]
            
            image_mask = batch['masks'][:, :N]
            smpl_mask = batch['smpl_mask'][:, :N]

            mask = image_mask * smpl_mask # take intersection of masks
            
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None
            
            vc_pm_loss = loss_fn(
                pred_vc,
                gt_vc_smpl_pm,
                mask,
                confidence
            )

            vc_pm_loss *= self.cfg.LOSS.VC_PM_LOSS_WEIGHT
            loss_dict['vc_pm_loss'] = vc_pm_loss
            total_loss = total_loss + vc_pm_loss


        if "vc_init" in predictions and "vc_smpl_maps" in batch:
            assert dataset_name == 'THuman'
            # loss_fn = self.vc_pm_asap_loss
            loss_fn = self.vc_pm_asap_loss 
            
            pred_vc = predictions['vc_init']
            gt_vc_smpl_pm = batch['vc_smpl_maps'][:, :N]
            
            image_mask = batch['masks'][:, :N]
            smpl_mask = batch['smpl_mask'][:, :N]

            mask = image_mask * smpl_mask # take intersection of masks
            
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None
            
            vc_pm_loss = loss_fn(
                pred_vc,
                gt_vc_smpl_pm,
                mask,
                confidence
            )

            vc_pm_loss *= self.cfg.LOSS.VC_PM_LOSS_WEIGHT
            loss_dict['vc_pm_loss'] = vc_pm_loss
            total_loss = total_loss + vc_pm_loss


        if "w" in predictions and "smpl_w_maps" in batch:
            pred_w = predictions['w']
            gt_w = batch['smpl_w_maps'][:, :N]
            mask = batch['masks'][:, :N]
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None

            w_loss = self.skinning_weight_loss(
                gt_w, 
                pred_w, 
                mask,
                confidence
            )

            w_loss *= self.cfg.LOSS.W_REGULARISER_WEIGHT
            loss_dict['w_loss'] = w_loss
            total_loss = total_loss + w_loss

        if "vp_init" in predictions and "vp" in batch:
            gt_vp = batch['vp']
            pred_vp = predictions['vp_init']
            mask = batch['masks']
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None

            mask = rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            if confidence is not None:
                confidence = rearrange(confidence[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')

            vp_loss = self.posed_chamfer_loss(
                gt_vp,
                pred_vp, 
                mask,
                confidence
            ) 
            vp_loss *= self.cfg.LOSS.VP_INIT_CHAMFER_LOSS_WEIGHT
            loss_dict['vp_init_chamfer_loss'] = vp_loss
            total_loss = total_loss + vp_loss


        if "vp" in predictions and "vp" in batch:
            gt_vp = batch['vp']
            pred_vp = predictions['vp']
            mask = batch['masks']

            # confidence = predictions['dvc_conf'] if "dvc_conf" in predictions else None
            # mask = rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            # if confidence is not None:
            #     confidence = rearrange(confidence, 'b k n h w -> (b k) (n h w)')
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None
            mask = rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            if confidence is not None:
                confidence = rearrange(confidence[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')

            vp_loss = self.posed_chamfer_loss(
                gt_vp,
                pred_vp, 
                mask,
                confidence
            ) 
            vp_loss *= self.cfg.LOSS.VP_CHAMFER_LOSS_WEIGHT
            loss_dict['vp_chamfer_loss'] = vp_loss
            total_loss = total_loss + vp_loss
  

        loss_dict['total_loss'] = total_loss

        # for k, v in loss_dict.items():
        #     print(k, v.item())
        # print("total_loss", total_loss.item())
        # import ipdb; ipdb.set_trace()
        
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

        mask = mask.squeeze(-1).bool()  # (B, V2)
        if (mask.sum(dim=1) == 0).any():
            print("Zero valid points after masking, returning 0 loss")
            return torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)
        
        x_pred_list = [x_pred[b][mask[b]] for b in range(x_pred.shape[0])]
        conf_list = [confidence[b][mask[b]] for b in range(x_pred.shape[0])] if confidence is not None else None
        log_conf_list = [self.get_conf_log(conf_list[b])[-1] for b in range(len(conf_list))] if confidence is not None else None

        x_pred_ptclds = Pointclouds(points=x_pred_list)


        loss, _ = chamfer_distance(
            x_pred_ptclds, x_gt, 
            batch_reduction=None, 
            point_reduction=None
        )
        loss_pred2gt = loss[0]
        loss_gt2pred = loss[1]

        loss_pred2gt_list = []
        for b in range(x_pred.shape[0]):
            loss_pred2gt_list.append(loss_pred2gt[b][:mask[b].sum()])
        loss_pred2gt = torch.cat(loss_pred2gt_list, dim=0)

        if confidence is not None:
            conf = torch.cat(conf_list, dim=0)
            log_conf = torch.cat(log_conf_list, dim=0)
            loss_pred2gt = loss_pred2gt * conf - self.alpha * log_conf
            # loss_gt2pred *= 10000.        

        loss_pred2gt = filter_by_quantile(loss_pred2gt, 0.98)

        final_loss = loss_pred2gt.mean() + loss_gt2pred.mean()

        return final_loss 
    


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

        if mask.sum() == 0:
            print("Mask is all zeros, returning 0 loss")
            return torch.tensor(0.0, device=loss.device, dtype=torch.float32)

        final_loss = loss.sum() / (mask.sum() + 1e-6)
        return final_loss
    

class ASAPLoss(nn.Module):
    """
    As SMPL as possible loss
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, x, y, mask=None, uncertainty=None):
        threshold = 0.03

        norm = torch.norm(x - y, dim=-1)
        norm_mask = norm > threshold
        loss = torch.where(norm_mask, norm, torch.zeros_like(norm))
        

        if uncertainty is not None:
            conf, log_conf = self.get_conf_log(uncertainty)
            loss = loss * conf - self.alpha * log_conf

        if mask is not None:
            full_mask = norm_mask * mask
        else:
            full_mask = norm_mask

        loss = loss * full_mask

        # loss = check_and_fix_inf_nan(loss, 'asap_loss')
        if full_mask.sum() == 0:
            print("Full mask is all zeros, returning 0 loss")
            return torch.tensor(0.0, device=loss.device, dtype=torch.float32)

        return loss.sum() / (full_mask.sum() + 1e-6)




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
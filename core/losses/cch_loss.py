import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F

from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds

from core.utils.general import check_and_fix_inf_nan
from core.utils.loss_utils import filter_by_quantile


class CCHLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.posed_chamfer_loss = MaskedUncertaintyChamferLoss(cfg=cfg)
        self.canonical_chamfer_loss = MaskedUncertaintyChamferLoss(cfg=cfg)

        self.vc_pm_l2_loss = MaskedUncertaintyL2Loss()
        self.vc_pm_asap_loss = ASAPLoss()

        self.skinning_weight_loss = MaskedUncertaintyL2Loss(scale=False)
        self.dvc_loss = MaskedUncertaintyL2Loss()

        self.debug_chamfer_loss = DebugChamferLoss()



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
            if self.cfg.LOSS.USE_ASAP_LOSS_D4DRESS:
                loss_fn = self.vc_pm_asap_loss
            else:
                loss_fn = self.vc_pm_l2_loss

                
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

            if self.cfg.LOSS.USE_VC_NORMALS:
                loss_grad = gradient_loss_multi_scale_wrapper(
                    rearrange(pred_vc, 'b n h w c -> (b n) h w c'),
                    rearrange(gt_vc_smpl_pm, 'b n h w c -> (b n) h w c'),
                    rearrange(mask.bool(), 'b n h w -> (b n) h w'),
                    gradient_loss_fn=normal_loss,
                    scales=3,
                    conf=rearrange(confidence, 'b n h w -> (b n) h w'),
                )
                loss_grad *= self.cfg.LOSS.VC_NORMAL_LOSS_WEIGHT
                loss_dict['vc_normal_loss'] = loss_grad
                total_loss = total_loss + loss_grad


        if "vc_init" in predictions and "vc_smpl_maps" in batch:
            assert dataset_name == 'THuman'
            if self.cfg.LOSS.USE_ASAP_LOSS_THUMAN:
                loss_fn = self.vc_pm_asap_loss 
            else:
                loss_fn = self.vc_pm_l2_loss
            
            pred_vc = predictions['vc_init'] # (B, N, H, W, 3)
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


            if self.cfg.LOSS.USE_VC_NORMALS:
                loss_grad = gradient_loss_multi_scale_wrapper(
                    rearrange(pred_vc, 'b n h w c -> (b n) h w c'),
                    rearrange(gt_vc_smpl_pm, 'b n h w c -> (b n) h w c'),
                    rearrange(mask.bool(), 'b n h w -> (b n) h w'),
                    gradient_loss_fn=normal_loss,
                    scales=3,
                    conf=rearrange(confidence, 'b n h w -> (b n) h w'),
                )
                loss_grad *= self.cfg.LOSS.VC_NORMAL_LOSS_WEIGHT
                loss_dict['vc_normal_loss'] = loss_grad
                total_loss = total_loss + loss_grad


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


            # confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None
            # _, debug_loss_pred2gt_conf, debug_loss_pred2gt, debug_loss_gt2pred = self.debug_chamfer_loss(
            #     gt_vp,
            #     rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c'), 
            #     rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)'),
            #     rearrange(confidence.unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            # )
            # # loss_dict['debug_vp_loss'] = rearrange(debug_vp_loss, '(b k) (n h w) -> b k n h w', b=B, k=K, n=N, h=H, w=W)
            # loss_dict['debug_loss_pred2gt_conf'] = rearrange(debug_loss_pred2gt_conf, '(b k) (n h w) -> b k n h w', b=B, k=K, n=N, h=H, w=W)
            # loss_dict['debug_loss_pred2gt'] = rearrange(debug_loss_pred2gt, '(b k) (n h w) -> b k n h w', b=B, k=K, n=N, h=H, w=W)
            # loss_dict['debug_loss_gt2pred'] = debug_loss_gt2pred

            # confidence = predictions['dvc_conf'] if "dvc_conf" in predictions else None
            # mask = rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            # if confidence is not None:
            #     confidence = rearrange(confidence, 'b k n h w -> (b k) (n h w)')
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None
            mask = rearrange(mask[:, :N].unsqueeze(1).repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')
            if confidence is not None:
                confidence = rearrange(confidence[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')

            if self.cfg.LOSS.VP_CHAMFER_LOSS_USE_UNCERTAINTY:
                pass
            else:
                confidence = None

            vp_loss = self.posed_chamfer_loss(
                gt_vp,
                pred_vp, 
                mask,
                confidence
            ) 
            vp_loss *= self.cfg.LOSS.VP_CHAMFER_LOSS_WEIGHT
            loss_dict['vp_chamfer_loss'] = vp_loss
            total_loss = total_loss + vp_loss


        if "vp" in predictions and "scan_pointmaps" in batch and self.cfg.LOSS.USE_NORMALS:
            pred = predictions['vp'] # (B, K, N, H, W, 3)
            gt = batch['scan_pointmaps'] # (B, K, H, W, 3)
            mask = batch['masks'].bool()
            confidence = predictions['vc_init_conf'] if "vc_init_conf" in predictions else None # (B, N, H, W)

            pred = torch.stack([pred[:, i, i] for i in range(4)], dim=1)
            pred = rearrange(pred, 'b n h w c -> (b n) h w c')
            gt = rearrange(gt[:, :N], 'b n h w c -> (b n) h w c')
            mask = rearrange(mask[:, :N], 'b n h w -> (b n) h w')
            confidence = rearrange(confidence, 'b n h w -> (b n) h w')


            loss_grad = gradient_loss_multi_scale_wrapper(
                pred,
                gt,
                mask,
                gradient_loss_fn=normal_loss,
                scales=3,
                conf=confidence,
            )
            loss_grad *= self.cfg.LOSS.VP_NORMAL_LOSS_WEIGHT
            loss_dict['vp_normal_loss'] = loss_grad
            total_loss = total_loss + loss_grad


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
    def __init__(self, alpha=1.0, cfg=None):
        super().__init__()
        self.alpha = alpha
        self.cfg = cfg

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
        loss_pred2gt = torch.sqrt(loss[0]) * 100. # convert to cm^2
        loss_gt2pred = torch.sqrt(loss[1]) * 100. # convert to cm^2

        loss_pred2gt_list = []
        for b in range(x_pred.shape[0]):
            loss_pred2gt_list.append(loss_pred2gt[b][:mask[b].sum()])
        loss_pred2gt = torch.cat(loss_pred2gt_list, dim=0)

        if confidence is not None:
            conf = torch.cat(conf_list, dim=0)
            log_conf = torch.cat(log_conf_list, dim=0)
            loss_pred2gt = loss_pred2gt * conf - self.alpha * log_conf
            loss_gt2pred *= self.cfg.LOSS.SCALE_GT2PRED

        loss_pred2gt = filter_by_quantile(loss_pred2gt, 0.98)

        final_loss = loss_pred2gt.mean() + loss_gt2pred.mean()

        return final_loss
    


class DebugChamferLoss(nn.Module):
    """
    Masked chamfer loss

    Args:
        x_gt: (B, V1, 3)
        x_pred: (B, V2, 3)
        mask: (B, V2, 1)
    """
    def __init__(self, alpha=1.0, cfg=None):
        super().__init__()
        self.alpha = alpha
        self.cfg = cfg

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, x_gt, x_pred, mask, confidence=None):
        assert x_pred.shape[:2] == mask.shape[:2]

        x_pred = x_pred * mask.unsqueeze(-1)

        loss, _ = chamfer_distance(
            x_pred, x_gt, 
            batch_reduction=None, 
            point_reduction=None
        )
        loss_pred2gt = torch.sqrt(loss[0]) * 100. # convert to cm^2
        loss_gt2pred = torch.sqrt(loss[1]) * 100. # convert to cm^2

        if confidence is not None:
            loss_pred2gt_conf = loss_pred2gt * confidence - self.alpha * torch.log(confidence)
            loss_pred2gt = loss_pred2gt * mask 

        loss_pred2gt_conf = loss_pred2gt_conf * mask

        # final_loss = (loss_pred2gt_conf + loss_gt2pred) / 2

        # final_loss = loss_pred2gt.mean() + loss_gt2pred.mean()
        return None, loss_pred2gt_conf, loss_pred2gt, loss_gt2pred


class MaskedUncertaintyL2Loss(nn.Module):
    """
    Masked L2 loss weighted by uncertainty 
    """
    def __init__(self, alpha=1.0, scale=True):
        super().__init__()
        self.alpha = alpha
        self.scale = scale

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, x, y, mask=None, uncertainty=None):

        loss = torch.norm(x - y, dim=-1)
        if self.scale:
            loss = loss * 100. # convert to cm^2

        if uncertainty is not None:
            conf, log_conf = self.get_conf_log(uncertainty)
            loss = loss * conf - self.alpha * log_conf

        if mask is not None:
            loss = loss * mask

        return loss.sum() / (mask.sum() + 1e-6)
    

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
        threshold = 0.02

        norm = torch.norm(x - y, dim=-1)
        norm_mask = norm > threshold
        loss = torch.where(norm_mask, norm, torch.zeros_like(norm))

        loss = loss * 100. # convert to cm^2

        if uncertainty is not None:
            conf, log_conf = self.get_conf_log(uncertainty)
            loss = loss * conf - self.alpha * log_conf

        if mask is not None:
            full_mask = norm_mask * mask
        else:
            full_mask = norm_mask

        loss = loss * full_mask

        return loss.sum() / (full_mask.sum() + 1e-6)




def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids



def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


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
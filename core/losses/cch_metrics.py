import torch
import pytorch_lightning as pl
from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds

class CCHMetrics(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.threshold = cfg.LOSS.CONFIDENCE_THRESHOLD

    def forward(self, predictions, batch):
        ret = {}

        B, N, H, W, _ = predictions['vc_init'].shape
        K = N 

        if "vc_conf" in predictions:
            confidence = predictions['vc_conf']
            confidence = confidence > self.threshold
        else:
            confidence = torch.ones_like(predictions['vc_init'])[..., 0].bool()

        assert confidence.shape == batch['masks'].shape


        if "vc_init" in predictions:
            gt_vc = Pointclouds(
                points=batch['template_mesh_verts']
            )
            pred_vc = predictions['vc_init']
            mask = batch['masks'] * confidence 

            pred_vc = rearrange(pred_vc, 'b n h w c -> b (n h w) c')
            mask = rearrange(mask, 'b n h w -> b (n h w)')

            vc_cfd, _, _ = self.masked_metric_cfd(gt_vc, pred_vc, mask)
            ret['vc_cfd'] = vc_cfd
        
        if "vp_init" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp_init']
            mask = batch['masks'] * confidence 

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_cfd, _, _ = self.masked_metric_cfd(gt_vp, pred_vp, mask)
            ret['vp_init_cfd'] = vp_cfd

        if "vp" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp']
            mask = batch['masks'] * confidence 

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_cfd, _, _ = self.masked_metric_cfd(gt_vp, pred_vp, mask)
            ret['vp_cfd'] = vp_cfd 
            
            
        return ret



    def masked_metric_cfd(self, x_gt, x_pred, mask):
        cfd_ret, _ = chamfer_distance(
            x_pred, x_gt, 
            batch_reduction=None, 
            point_reduction=None
        )

        loss_pred2gt = cfd_ret[0]
        loss_gt2pred = cfd_ret[1]

        masked_loss_pred2gt = loss_pred2gt * mask

        dist_pred2gt = torch.sum(torch.sqrt(masked_loss_pred2gt)) / (mask.sum() + 1e-6) * 100.0
        dist_gt2pred = torch.sqrt(loss_gt2pred).mean() * 100.0


        return (dist_pred2gt+dist_gt2pred) / 2, dist_pred2gt, dist_gt2pred
    

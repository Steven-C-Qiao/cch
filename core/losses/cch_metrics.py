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
        K = 5

        if "vc_init_conf" in predictions:
            confidence = predictions['vc_init_conf']
            confidence = confidence > self.threshold
        else:
            confidence = torch.ones_like(predictions['vc_init'])[..., 0].bool()

        assert confidence.shape == batch['masks'][:, :N].shape


        # if "vc_init" in predictions:
        #     gt_vc = Pointclouds(
        #         points=batch['template_mesh_verts']
        #     )
        #     pred_vc = predictions['vc_init']
        #     mask = batch['masks'][:, :N] * confidence 

        #     pred_vc = rearrange(pred_vc, 'b n h w c -> b (n h w) c')
        #     mask = rearrange(mask, 'b n h w -> b (n h w)')

        #     vc_cfd, _, _ = self.masked_metric_cfd(gt_vc, pred_vc, mask)
        #     ret['vc_cfd'] = vc_cfd

        
        if "vp_init" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp_init']
            mask = batch['masks'][:, :N] * confidence 

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_cfd, _, _ = self.masked_metric_cfd(gt_vp, pred_vp, mask)
            ret['vp_init_cfd'] = vp_cfd


            # An extra frame in the end 
            extra_gt_vp = Pointclouds(batch['vp_ptcld'].points_list()[4::K])
            extra_pred_vp = rearrange(predictions['vp_init'][:, -1], 'b n h w c -> b (n h w) c')
            mask = rearrange((batch['masks'][:, :N] * confidence), 'b n h w -> b (n h w)')

            extra_vp_cfd, _, _ = self.masked_metric_cfd(extra_gt_vp, extra_pred_vp, mask)
            ret['extra_vp_init_cfd'] = extra_vp_cfd


        if "vp" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp']
            mask = batch['masks'][:, :N] * confidence 

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_cfd, _, _ = self.masked_metric_cfd(gt_vp, pred_vp, mask)
            ret['vp_cfd'] = vp_cfd 

            # An extra frame in the end 
            extra_gt_vp = Pointclouds(batch['vp_ptcld'].points_list()[4::K])
            extra_pred_vp = rearrange(predictions['vp'][:, -1], 'b n h w c -> b (n h w) c')
            mask = rearrange((batch['masks'][:, :N] * confidence), 'b n h w -> b (n h w)')

            extra_vp_cfd, _, _ = self.masked_metric_cfd(extra_gt_vp, extra_pred_vp, mask)
            ret['extra_vp_cfd'] = extra_vp_cfd

            
            
        return ret



    def masked_metric_cfd(self, x_gt, x_pred, mask):
        # Filter x_pred using mask to reduce computation
        mask_flat = mask.bool()  # (B, V2)
        
        x_pred_list = [x_pred[b][mask_flat[b]] for b in range(x_pred.shape[0])]
        
        # Convert list to Pointclouds object for chamfer_distance compatibility
        x_pred_ptclds = Pointclouds(points=x_pred_list)
        
        cfd_ret, _ = chamfer_distance(
            x_pred_ptclds, x_gt, 
            batch_reduction=None, 
            point_reduction=None
        )

        loss_pred2gt = cfd_ret[0]  # Shape: (B, P) where P is max points across batch
        loss_gt2pred = cfd_ret[1]

        # Handle padding: only sum over valid (non-padded) points
        valid_points_per_batch = [len(x_pred_list[b]) for b in range(len(x_pred_list))]
        total_valid_points = sum(valid_points_per_batch)
        
        # Create mask for valid points in the padded tensor
        valid_mask = torch.zeros_like(loss_pred2gt)
        for b, num_valid in enumerate(valid_points_per_batch):
            if num_valid > 0:
                valid_mask[b, :num_valid] = 1.0
        
        masked_loss_pred2gt = loss_pred2gt * valid_mask

        dist_pred2gt = torch.sum(torch.sqrt(masked_loss_pred2gt)) / (total_valid_points + 1e-6) * 100.0
        dist_gt2pred = torch.sqrt(loss_gt2pred).mean() * 100.0

        return (dist_pred2gt+dist_gt2pred) / 2, dist_pred2gt, dist_gt2pred
    

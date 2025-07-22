import torch
import pytorch_lightning as pl
from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds

class CCHMetrics(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, predictions, batch):
        ret = {}

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

            vc_cfd, _, _ = self.masked_metric_cfd(gt_vc, pred_vc, mask)
            ret['vc_cfd'] = vc_cfd
        
        if "vp_init" in predictions:
            gt_vp = batch['vp_ptcld']
            pred_vp = predictions['vp_init']
            mask = batch['masks']

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
    


    # if conf is not None:
        #     conf_threshold = 0.08
        #     conf_mask = (1/conf) < conf_threshold
        #     full_mask = (mask * conf_mask).bool()
        # else:
        #     full_mask = mask.bool()

        # show_prog_bar = (split == 'train')
        # # ----------------------- vc -----------------------
        # if vc is not None and vc_pred is not None:
        #     vc_pm_dist = torch.norm(vc - vc_pred, dim=-1) * full_mask 
        #     vc_pm_dist = vc_pm_dist.sum() / (full_mask.sum() + 1e-6) * 100.0
        #     self.log(f'{split}_vc_pm_dist', vc_pm_dist,  on_step=True, on_epoch=True, prog_bar=show_prog_bar, sync_dist=True, rank_zero_only=True)


        # # ----------------------- vp -----------------------
        # full_vp_mask = rearrange(full_mask, '(b n) (h w) -> b n h w', 
        #                             b=B, n=N, h=self.image_size, w=self.image_size)
        # full_vp_mask = full_vp_mask[:, None].repeat(1, N, 1, 1, 1)
        # full_vp_mask = rearrange(full_vp_mask, 'b k n h w -> (b k) (n h w)')

        # vp_dist_squared, _ = chamfer_distance(vp_pred, vp, batch_reduction=None, point_reduction=None)
        # vpp2vp_dist = torch.sqrt(vp_dist_squared[0])
        # masked_vpp2vp_dist = vpp2vp_dist * full_vp_mask
        # vpp2vp_dist = masked_vpp2vp_dist.sum() / (full_vp_mask.sum() + 1e-6) * 100.0

        # vp2vpp_dist = torch.sqrt(vp_dist_squared[1])
        # vp2vpp_dist = vp2vpp_dist.mean() * 100.0


        # self.log(f'{split}_vpp2vp_cfd', vpp2vp_dist, on_step=True, on_epoch=True, prog_bar=show_prog_bar, sync_dist=True, rank_zero_only=True)
        # self.log(f'{split}_vp2vpp_cfd', vp2vpp_dist, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)

        # return {
        #     # 'vc_pm_dist': vc_pm_dist,
        #     'vpp2vp_cfd': vpp2vp_dist,
        #     'vp2vpp_cfd': vp2vpp_dist,
        # }
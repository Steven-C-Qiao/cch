import torch 
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds

from einops import rearrange

from core.heads.dpt_head import DPTHead
from core.models.cch_aggregator import Aggregator
from core.heads.pose_corrective_head import PoseCorrectiveHead
from core.utils.general_lbs import general_lbs
from core.utils.diffpointrend import PointCloudRenderer


class CCH(nn.Module):
    def __init__(self, cfg, smpl_model, img_size=224, patch_size=14, embed_dim=384):
        """
        Given a batch of normal images, predict pixel-aligned canonical space position and uv coordinates
        """
        super(CCH, self).__init__()
        self.smpl_model = smpl_model
        self.parents = smpl_model.parents
        self.cfg = cfg

        self.model_skinning_weights = cfg.MODEL.SKINNING_WEIGHTS
        self.model_pose_correctives = cfg.MODEL.POSE_CORRECTIVES

        # self.renderer = PointCloudRenderer(image_size=(img_size, img_size))

        # self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv")
        # self.canonical_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        # if self.model_skinning_weights:
        #     self.skinning_head = DPTHead(dim_in=2 * embed_dim, output_dim=25, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3)
        # else:
        #     self.skinning_head = None

        if self.model_pose_correctives:
            self.pose_correctives_aggregator = Aggregator(
                img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv", input_channels=6
            )
            # per-frame pose correctives and uncertainty
            # self.pose_correctives_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3) 
            self.pose_correctives_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1")
    
            # self.pose_correctives_aggregator = Aggregator(
            #     img_size=img_size, patch_size=patch_size, embed_dim=64, mlp_ratio=2.0, num_heads=2,
            #     patch_embed="conv", input_channels=6
            # )
            # self.pose_correctives_head = DPTHead(dim_in=2 * 64, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1",
            #                                      features=16,out_channels=[12, 12, 12, 12])



    def forward(self, images, pose=None, joints=None, w_smpl=None, mask=None, R=None, T=None, gt_vc=None):
        """
        return:
            vc: canonical pointmaps: (B, N, H, W, 3)
            vc_conf: vc confidence maps: (B, N, H, W, 1)
            vp: posed vertices from vc: (B, N, H, W, 3)
            w: skinning weights: (B, N, H, W, 25)
            w_conf: skinning weights confidence maps: (B, N, H, W, 1) or None 
            dvc: pose correctives: (B, N, H, W, 3)
        """

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        vc = gt_vc
        w = w_smpl


        vp, _ = general_lbs(
            vc=rearrange(vc, 'b n h w c -> (b n) (h w) c'),
            pose=rearrange(pose, 'b n c -> (b n) c'),
            lbs_weights=rearrange(w, 'b n h w j -> (b n) (h w) j'),
            J=joints.repeat_interleave(pose.shape[1], dim=0),
            parents=self.smpl_model.parents 
        )
        vp = rearrange(vp, '(b n) (h w) c -> b n h w c', h=H, w=W, b=B, n=N)
        vp = vp * mask.unsqueeze(-1)



        full_cond = torch.cat([rearrange(vc, 'b n h w c -> b n c h w'), 
                               rearrange(vp, 'b n h w c -> b n c h w')], dim=-3)
        aggregated_tokens_list_cond, patch_start_idx = self.pose_correctives_aggregator(full_cond)
        dvc, _ = self.pose_correctives_head(aggregated_tokens_list_cond, images, patch_start_idx=patch_start_idx)
        dvc = (torch.sigmoid(dvc) - 0.5) * 0.2
        vc = vc + dvc

        pred = {
            'vc': vc, 
            'vc_conf': None,
            'vp': vp,
            'w': w,
            'w_conf': None,
            'dvc': dvc, 
        }

        return pred 



    def count_parameters(self):
        """
        Count number of trainable parameters in aggregator and warper
        """
        aggregator_params = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
        warper_params = sum(p.numel() for p in self.warper.parameters() if p.requires_grad)
        
        print(f'Aggregator parameters: {aggregator_params:,}')
        print(f'Warper parameters: {warper_params:,}')
        print(f'Total parameters: {aggregator_params + warper_params:,}')
        
        return {
            'aggregator': aggregator_params,
            'warper': warper_params,
            'total': aggregator_params + warper_params
        }
    

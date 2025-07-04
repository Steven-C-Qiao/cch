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
        self.model_pose_blendshapes = cfg.MODEL.POSE_BLENDSHAPES

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv")
        self.canonical_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        if self.model_skinning_weights:
            self.skinning_head = DPTHead(dim_in=2 * embed_dim, output_dim=25, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3)

        if self.model_pose_blendshapes:
            self.pose_blendshapes_aggregator = Aggregator(
                img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv", input_channels=6
            )
            # per-frame pose correctives and uncertainty
            # self.pose_correctives_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3) 
            self.pose_blendshapes_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1")
    
            # self.pose_correctives_aggregator = Aggregator(
            #     img_size=img_size, patch_size=patch_size, embed_dim=64, mlp_ratio=2.0, num_heads=2,
            #     patch_embed="conv", input_channels=6
            # )
            # self.pose_correctives_head = DPTHead(dim_in=2 * 64, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1",
            #                                      features=16,out_channels=[12, 12, 12, 12])



    def forward(self, images, pose=None, joints=None, w_smpl=None, mask=None):
        """
        Inputs:
        Returns:
            vc: canonical pointmaps: (B, N, H, W, 3)
            vc_conf: vc confidence maps: (B, N, H, W, 1)
            vp: posed vertices from vc: (B, N, H, W, 3)
            w: skinning weights: (B, N, H, W, 25)
            w_conf: skinning weights confidence maps: (B, N, H, W, 1) 
            dvc: pose correctives: (B, N, H, W, 3) 
        """

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images) 

        vc_init, vc_conf_init = self.canonical_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx)
        vc_init = torch.clamp(vc_init, -2, 2)


        if self.model_skinning_weights:
            w, w_conf = self.skinning_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx, 
                                           additional_conditioning=vc_init) # rearrange(vc, 'b n h w c -> (b n) c h w'))
            w = F.softmax(w, dim=-1)
        else:
            w, w_conf = w_smpl, None


        if self.model_pose_blendshapes:
            # """
            # Generate pose condition as the posed colormap 
            # """

            # vp_init, _ = general_lbs(
            #     vc=rearrange(vc_init, 'b n h w c -> (b n) (h w) c'),
            #     pose=rearrange(pose, 'b n c -> (b n) c'),
            #     lbs_weights=rearrange(w, 'b n h w j -> (b n) (h w) j'),
            #     J=joints.repeat_interleave(pose.shape[1], dim=0),
            #     parents=self.parents 
            # )
            # vp_init = rearrange(vp_init, '(b n) (h w) c -> b n h w c', b=B, n=N, h=H, w=W)
            # vp_init = vp_init * mask.unsqueeze(-1)
            
            # pose_correctives_head_input = torch.cat([rearrange(vc_init, 'b n h w c -> b n c h w'),
            #                                          rearrange(vp_init, 'b n h w c -> b n c h w')], dim=-3)

            # pose_correctives_aggregated_tokens_list, _ = self.pose_correctives_aggregator(pose_correctives_head_input)
            # dvc, _ = self.pose_correctives_head(pose_correctives_aggregated_tokens_list, images, patch_start_idx=patch_start_idx)
            # dvc = (torch.sigmoid(dvc) - 0.5) * 0.2 # limit the update to [-0.1, 0.1]
            
            # vc = vc_init + dvc

            vp_init, dvc = self._forward_pose_blendshapes(vc_init, pose, w, joints, mask, images) # (B, N, N, H, W, 3), (B, N, N, H, W, 3)

            vc = vc_init[:, None] + dvc

        else:
            vp_init, dvc = None, None

        pred = {
            'vc_init_pred': vc_init,
            'vc_conf_init_pred': vc_conf_init,
            'vp_init_pred': vp_init,
            'w_pred': w,
            'w_conf_pred': w_conf,
            'vc_pred': vc,
            'dvc_pred': dvc
        }

        return pred 
    

    def _forward_pose_blendshapes(self, vc, pose, w, joints, mask, images):
        """
        Implement the full blend shape network. 
        Given each pose, regress full blend shapes for all canonical points.

        Args:
            vc: (B, N, H, W, 3)
            pose: (B, N, 72)

        Returns:
            vp_init: (B, N, N, H, W, 3)
            dvc: (B, N, N, H, W, 3)
        """

        B, N, H, W = vc.shape[:4]

        vc_expanded = vc.unsqueeze(1).repeat(1, N, 1, 1, 1, 1) # (B, N, N, H, W, 3)
        pose_expanded = pose.unsqueeze(1).repeat(1, N, 1, 1) # (B, N, N, 72)
        w_expanded = w.unsqueeze(1).repeat(1, N, 1, 1, 1, 1) # (B, N, N, H, W, 25)
        
        vp_init, _ = general_lbs(
            vc=rearrange(vc_expanded, 'b k n h w c -> (b k n) (h w) c'),
            pose=rearrange(pose_expanded, 'b k n c -> (b k n) c'),
            lbs_weights=rearrange(w_expanded, 'b k n h w j -> (b k n) (h w) j'),
            J=joints.repeat_interleave(N * N, dim=0),
            parents=self.parents 
        )
        # These are the posed colormaps for each pose for all canonical points
        vp_init = rearrange(vp_init, '(b k n) (h w) c -> b k n h w c', b=B, n=N, k=N, h=H, w=W)
        vp_init = vp_init * mask[:, None, ..., None] # expand N and channel dim 
        
        pose_blendshapes_aggregator_input = torch.cat([rearrange(vc_expanded, 'b k n h w c -> (b k) n c h w'),
                                                       rearrange(vp_init, 'b k n h w c -> (b k) n c h w')], dim=-3).contiguous()
        
        pose_blendshapes_aggregated_tokens_list, patch_start_idx = self.pose_blendshapes_aggregator(pose_blendshapes_aggregator_input)
        
        dvc, _ = self.pose_blendshapes_head(pose_blendshapes_aggregated_tokens_list, images.repeat_interleave(N, dim=0), patch_start_idx=patch_start_idx)
        dvc = (torch.sigmoid(dvc) - 0.5) * 0.2 # limit the update to [-0.1, 0.1]

        dvc = rearrange(dvc, '(b k) n h w c -> b k n h w c', b=B, k=N)

        return vp_init, dvc


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
    

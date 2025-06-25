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

        self.renderer = PointCloudRenderer(image_size=(img_size, img_size))

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv")
        self.canonical_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        if self.model_skinning_weights:
            self.skinning_head = DPTHead(dim_in=2 * embed_dim, output_dim=25, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3)

        if self.model_pose_correctives:
            self.pose_correctives_aggregator = Aggregator(
                img_size=img_size, patch_size=patch_size, embed_dim=64, mlp_ratio=2.0, num_heads=2,
                patch_embed="conv", input_channels=6
            )
            # per-frame pose correctives and uncertainty
            # self.pose_correctives_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3) 
            self.pose_correctives_head = DPTHead(dim_in=2 * 64, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1")


    def forward(self, images, pose=None, joints=None, w_smpl=None, mask=None, R=None, T=None):
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

        vc, vc_conf = self.canonical_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx)
        vc = torch.clamp(vc, -2, 2)


        if self.model_skinning_weights:
            w, w_conf = self.skinning_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx, 
                                           additional_conditioning=vc) # rearrange(vc, 'b n h w c -> (b n) c h w'))
            w = F.softmax(w, dim=-1)
        else:
            w, w_conf = w_smpl, None


        if self.model_pose_correctives:
            """
            To predict pose correctives, predict pointmap updates conditioned on pose.
            To condition on pose, use LBS to generate the posed pointmaps, which is coarsely pixel-aligned with vc pointmaps.
            """

            vp_init, _ = general_lbs(
                vc=rearrange(vc, 'b n h w c -> (b n) (h w) c'),
                pose=rearrange(pose, 'b n c -> (b n) c'),
                lbs_weights=rearrange(w, 'b n h w j -> (b n) (h w) j'),
                J=joints.repeat_interleave(pose.shape[1], dim=0),
                parents=self.parents 
            )
            vp_init = rearrange(vp_init, '(b n) (h w) c -> b n h w c', b=B, n=N, h=H, w=W)
            
            pose_correctives_head_input = torch.cat([vc, vp_init], dim=-3)

            pose_correctives_aggregated_tokens_list, _ = self.pose_correctives_aggregator(pose_correctives_head_input)
            dvc, _ = self.pose_correctives_head(pose_correctives_aggregated_tokens_list, images, patch_start_idx=patch_start_idx)
            dvc = (torch.sigmoid(dvc) - 0.5) * 0.2 # limit the update to [-0.1, 0.1]
            
            vc = vc + dvc

        else:
            vp_init, dvc = None, None

        # import ipdb; ipdb.set_trace()
            

        pred = {
            'vc_pred': vc,
            'vc_conf_pred': vc_conf,
            'vp_init_pred': vp_init,
            'w_pred': w,
            'w_conf_pred': w_conf,
            'dvc_pred': dvc
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
    

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
        else:
            self.skinning_head = None

        if self.model_pose_correctives:
            self.pose_correctives_aggregator = Aggregator(
                img_size=img_size, patch_size=patch_size, embed_dim=64, mlp_ratio=2.0, num_heads=2,
                patch_embed="conv", input_channels=6
            )
            # per-frame pose correctives and uncertainty
            # self.pose_correctives_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3) 
            self.pose_correctives_head = DPTHead(dim_in=2 * 64, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1")


    def forward(self, images, pose=None, joints=None, w_smpl=None, mask=None, R=None, T=None):

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images) 

        # with torch.cuda.amp.autocast(enabled=False):
        vc, vc_conf = self.canonical_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx)

        # add a clipping to vc for stability 
        vc = torch.clamp(vc, -2, 2)


        if self.model_skinning_weights:
            w, w_conf = self.skinning_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx, 
                                           additional_conditioning=vc) # rearrange(vc, 'b n h w c -> (b n) c h w'))
            w = F.softmax(w, dim=-1)
        else:
            w, w_conf = w_smpl, None




        if self.model_pose_correctives:
            """
            Render initial vp predictions without pose blendshapes using a pointcloud renderer
            use rendering and vc initial predictions to predict updates to vc initial.
            """
            vp, joints_posed = general_lbs(
                vc=rearrange(vc, 'b n h w c -> (b n) (h w) c'),
                pose=rearrange(pose, 'b n c -> (b n) c'),
                lbs_weights=rearrange(w, 'b n h w j -> (b n) (h w) j'),
                J=joints.repeat_interleave(pose.shape[1], dim=0),
                parents=self.parents 
            )
            # vp = rearrange(vp, '(b n) v c -> b n v c', b=B, n=N)
            mask = rearrange(mask, 'b n c h w -> (b n) (c h w)') # hw = v

            feat = vp.clone() # feats are 3d coordinates of vp, so vp

            masked_vp, masked_feat = [], []
            for i in range(B * N):
                masked_vp.append(vp[i, mask[i].bool()])
                masked_feat.append(feat[i, mask[i].bool()])

            
            pointclouds = Pointclouds(points=masked_vp, features=masked_feat)

            renderer_ret = self.renderer(pointclouds, R=R, T=T)
            vp_cond = renderer_ret['images']
            vp_cond_mask = renderer_ret['background_mask']
            vp_cond = rearrange(vp_cond, '(b n) h w c -> b n c h w', b=B, n=N)
            vc_cond = rearrange(vc, 'b n h w c -> b n c h w', b=B, n=N)

            # visualise a vp_cond 
            # import matplotlib.pyplot as plt
            # import numpy as np
            # vp_vis = vp_cond[0, 0, :, :, :]
            # vp_vis = vp_vis.cpu().numpy()
            # vp_vis = vp_vis.transpose(1, 2, 0)
            # vp_vis = (vp_vis + 1) / 2
            # vp_vis = vp_vis.clip(0, 1)
            # vp_vis = vp_vis * 255
            # vp_vis = vp_vis.astype(np.uint8)

            # plt.imshow(vp_vis)
            # plt.savefig('vp_cond.png')
            # plt.close()

            # import ipdb; ipdb.set_trace()


            full_cond = torch.cat([vc_cond, vp_cond], dim=-3)
            aggregated_tokens_list_cond, _ = self.pose_correctives_aggregator(full_cond)
            dvc, _ = self.pose_correctives_head(aggregated_tokens_list_cond, images, patch_start_idx=patch_start_idx)
            dvc = (torch.sigmoid(dvc) - 0.5) * 0.2
            vc = vc + dvc

            vp = rearrange(vp, '(b n) (h w) c -> b n (h w) c', h=H, w=W, b=B, n=N)
            vp_cond_mask = rearrange(vp_cond_mask, '(b n) h w -> b n h w', b=B, n=N)
        else:
            vp, dvc, vp_cond, vp_cond_mask = None, None, None, None

        # import ipdb; ipdb.set_trace()
            

        pred = {
            'vc': vc,
            'vc_conf': vc_conf,
            'vp': vp,
            'w': w,
            'w_conf': w_conf,
            'dvc': dvc,
            'vp_cond': vp_cond,
            'vp_cond_mask': vp_cond_mask,
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
    

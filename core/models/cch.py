import torch 
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from core.heads.dpt_head import DPTHead
from core.models.cch_aggregator import Aggregator
from core.heads.pose_corrective_head import PoseCorrectiveHead


class CCH(nn.Module):
    def __init__(self, cfg, smpl_model, img_size=224, patch_size=14, embed_dim=384, predict_smpl_topology=True):
        """
        Given a batch of normal images, predict pixel-aligned canonical space position and uv coordinates
        """
        super(CCH, self).__init__()
        self.smpl_model = smpl_model
        self.cfg = cfg

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv")
        self.canonical_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        if cfg.MODEL.SKINNING_WEIGHTS:
            self.skinning_head = DPTHead(dim_in=2 * embed_dim, output_dim=25, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3)
        else:
            self.skinning_head = None

        if cfg.MODEL.POSE_CORRECTIVES:
            # per-frame pose correctives and uncertainty
            # self.pose_correctives_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3) 
            self.pose_correctives_head = PoseCorrectiveHead(condition_dim=72)

        # if predict_smpl_topology:
        #     self.mesh_head = DPTHead(dim_in=2 * embed_dim, output_dim=25, activation="inv_log", conf_activation="expp1", feature_only=True)

        # self.count_parameters()



    def forward(self, images, pose=None):

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images) 

        # with torch.cuda.amp.autocast(enabled=False):
        vc, vc_conf = self.canonical_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx)

        # add a clipping to vc for stability 
        vc = torch.clamp(vc, -2, 2)


        if self.cfg.MODEL.SKINNING_WEIGHTS:
            w, w_conf = self.skinning_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx, 
                                           additional_conditioning=vc) # rearrange(vc, 'b n h w c -> (b n) c h w'))
            w = F.softmax(w, dim=-1)
        else:
            w, w_conf = None, None

        if self.cfg.MODEL.POSE_CORRECTIVES:
            # Given pose, and w, repose canonical points and render the posed position maps. This serves as the condition for the pose corrective head.
            # The pose corrective head will take in the posed position maps and the canonical position maps, and output the pose corrective.
            # The pose corrective is then added to the canonical position maps to get the final position maps.

            pose_correctives = self.pose_correctives_head(pose, vc)


            # pose_correctives, pose_correctives_conf = self.pose_correctives_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx, 
            #                                                                      additional_conditioning=vc)
            # pose_correctives = (torch.sigmoid(pose_correctives) - 0.5) * 0.2
        else:
            pose_correctives, pose_correctives_conf = None, None

        # if self.mesh_head is not None:
        #     mesh_pred = self.mesh_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx, 
        #                                additional_conditioning=vc)
        # else:
        #     mesh_pred = None


        # import ipdb; ipdb.set_trace()



        pred = {
            'vc': vc,
            'vc_conf': vc_conf,
            'w': w,
            'w_conf': w_conf,
            'dvc': pose_correctives,
            'dvc_conf': pose_correctives_conf
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
    

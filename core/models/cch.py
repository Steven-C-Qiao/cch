import torch 
import torch.nn as nn
import torch.nn.functional as F


from einops import rearrange

from core.heads.dpt_head import DPTHead
from core.models.cch_aggregator import Aggregator


class CCH(nn.Module):
    def __init__(self, cfg, smpl_model, img_size=224, patch_size=14, embed_dim=384):
        """
        Given a batch of normal images, predict pixel-aligned canonical space position and uv coordinates
        """
        super(CCH, self).__init__()
        self.smpl_model = smpl_model

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv")
        self.canonical_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        
        # use inv_log since now producing updates to the coarse_smpl_skinning_weights 
        self.skinning_head = DPTHead(dim_in=2 * embed_dim, output_dim=25, activation="inv_log", conf_activation="expp1")


        # self.count_parameters()



    def forward(self, images):

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images) 

        with torch.cuda.amp.autocast(enabled=False):
            vc, vc_conf = self.canonical_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx)
            w, w_conf = self.skinning_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx)

            # normalise skinning weights across joints 
            # No softmax now as using inv_log activation
            # w = F.softmax(w, dim=-1)

            # TODO: Pose blend shapes 

        pred = {
            'vc': vc,
            'vc_conf': vc_conf,
            'w': w,
            'w_conf': w_conf
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
    

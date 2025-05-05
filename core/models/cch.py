import torch 
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from core.backbone.vit import vit_base
from core.heads.dpt_head import DPTHead

class CCH(nn.Module):
    def __init__(self, cfg, smpl_model, embed_dim=768):
        """
        Given a batch of normal images, predict pixel-aligned canonical space position and uv coordinates
        """
        super(CCH, self).__init__()
        self.smpl_model = smpl_model

        self.backbone = vit_base()
        self.warper = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")



    def forward(self, batch, iters=2, epoch=0):
        # device = next(self.parameters()).device
        B, N, C_in, H, W = batch['normal_imgs'].shape

        normals = batch['normal_imgs']
        pose = batch['pose']
        betas = batch['shape']
        # underlying_smpl_mesh = self.smpl_model(body_pose=pose, shape=betas)

        feats = self.backbone(normals)
        vc, vc_conf = self.warper(feats, pose)

        # posed_verts = lbs(canonical_verts, pose, skinning_weights)

        pred = {
            'vc': vc,
            'vc_conf': vc_conf,
        }
        return pred 

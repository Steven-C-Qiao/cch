import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pytorch3d.structures import Pointclouds

from core.heads.dpt_head import DPTHead
from core.models.cch_aggregator import Aggregator
from core.utils.general_lbs import general_lbs


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
        self.model_pbs = cfg.MODEL.POSE_BLENDSHAPES

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="dinov2_vits14_reg")
        self.canonical_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        if self.model_skinning_weights:
            self.skinning_head = DPTHead(dim_in=2 * embed_dim, output_dim=25, activation="inv_log", conf_activation="expp1", additional_conditioning_dim=3)

        if self.model_pbs:
            self.pbs_aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed="conv", input_channels=6)
            self.pbs_head = DPTHead(dim_in=2 * embed_dim, output_dim= 3 + 1, activation="inv_log", conf_activation="expp1")
            
    def forward(self, images, pose=None, joints=None, w_smpl=None, mask=None):
        """
        Given surface normal images, predict the canonical human, and global pose blendshapes for each pose.
        Inputs:
            
        Returns:
            vc_init_pred: (B, N, H, W, 3)
            vc_conf: vc confidence maps: (B, N, H, W, 1)
            vc: canonical pointmaps: (B, K, N, H, W, 3): after correcting for the k-th pose

            w: skinning weights: (B, N, H, W, 25)
            w_conf: skinning weights confidence maps: (B, N, H, W, 1) 

            vp_init_pred: (B, K, N, H, W, 3): the K dimension are expanded, and are identical
            vp_init_cond_pred: (B, N, K, H, W, 3): Posed vertices for the k-th pose, conditioned on the k-th pose
            vp: (B, K, N, H, W, 3): Posed vertices for the k-th pose
            dvc: pose correctives: (B, K, N, H, W, 3) k-th pose blend shape for all N views 
        """
        ret = {}

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images) 

        vc_init, vc_conf_init = self.canonical_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx)
        vc_init = torch.clamp(vc_init, -2, 2)

        vc_init = vc_init * mask.unsqueeze(-1) # Mask background to 0, important for backward chamfer metrics

        ret['vc_init'] = vc_init
        ret['vc_conf_init'] = vc_conf_init




        if self.model_skinning_weights:
            w, w_conf = self.skinning_head(aggregated_tokens_list, images, patch_start_idx=patch_start_idx, 
                                           additional_conditioning=vc_init) # rearrange(vc, 'b n h w c -> (b n) c h w'))
            w = F.softmax(w, dim=-1)
        else:
            w, w_conf = w_smpl, None

        ret['w'] = w
        ret['w_conf'] = w_conf

        vc_init_expanded = vc_init.unsqueeze(1).repeat(1, N, 1, 1, 1, 1) # (B, K, N, H, W, 3)

        w_expanded = w.unsqueeze(1).repeat(1, N, 1, 1, 1, 1) # (B, K, N, H, W, 25)


        vp_init, J_init = general_lbs(
            vc=rearrange(vc_init_expanded, 'b k n h w c -> (b k) (n h w) c'),
            pose=rearrange(pose, 'b k c -> (b k) c'),
            lbs_weights=rearrange(w_expanded, 'b k n h w j -> (b k) (n h w) j'),
            J=rearrange(joints, 'b k j c -> (b k) j c'),
            parents=self.smpl_model.parents 
        )
        vp_init = rearrange(vp_init, '(b k) (n h w) c -> b k n h w c', b=B, k=N, n=N, h=H, w=W)
        J_init = rearrange(J_init, '(b k) j c -> b k j c', b=B, k=N)

        ret['vp_init'] = vp_init
        ret['J_init'] = J_init


        
        # Generate initial posed vertices without pose blendshapes
        # vc_expanded = vc_init.unsqueeze(1).repeat(1, N, 1, 1, 1, 1) # (B, K, N, H, W, 3)
        # pose_expanded = pose.unsqueeze(2).repeat(1, 1, N, 1) # (B, K, N, 69)
        # w_expanded = w.unsqueeze(1).repeat(1, N, 1, 1, 1, 1) # (B, K, N, H, W, 25)
        # mask = mask.unsqueeze(1).repeat(1, N, 1, 1, 1).unsqueeze(-1) # (B, K, N, H, W, 1)
        # joints = joints.unsqueeze(1).repeat(1, N, 1, 1, 1) # (B, K, N, 24, 3)

        # vp_init, J_init = general_lbs(
        #     vc=rearrange(vc_expanded, 'b k n h w c -> (b k n) (h w) c'),
        #     pose=rearrange(pose_expanded, 'b k n c -> (b k n) c'),
        #     lbs_weights=rearrange(w_expanded, 'b k n h w j -> (b k n) (h w) j'),
        #     J=rearrange(joints, 'b k n j c -> (b k n) j c'),
        #     parents=self.smpl_model.parents 
        # )
        # vp_init = rearrange(vp_init, '(b k n) (h w) c -> b k n h w c', b=B, n=N, k=N, h=H, w=W)
        # J_init = rearrange(J_init, '(b k n) j c -> b k n j c', b=B, n=N, k=N)
        # vp_init = vp_init * mask
        # ret['vp_init'] = vp_init
        # ret['J_init'] = J_init

        # Reshape to batch of point clouds and mask out invalid points
        # vp_init = rearrange(vp_init, 'b k n h w c -> (b k) (n h w) c')
        # mask = rearrange(mask.squeeze(-1), 'b k n h w -> (b k) (n h w)')
        # valid_points = [points[m] for points, m in zip(vp_init, mask)]
        
        # vp_init_ptcld = Pointclouds(points=valid_points)
        # ret['vp_init_ptcld'] = vp_init_ptcld


        if self.model_pbs:
            # Stack canonical and posed pointmaps to predict pose blendshapes
            pbs_aggregator_input = torch.cat([rearrange(vc_expanded, 'b k n h w c -> (b k) n c h w'),
                                              rearrange(vp_init, 'b k n h w c -> (b k) n c h w')], dim=-3).contiguous()
            
            pbs_aggregated_tokens_list, patch_start_idx = self.pbs_aggregator(pbs_aggregator_input)
            
            dvc, _ = self.pbs_head(pbs_aggregated_tokens_list, images.repeat_interleave(N, dim=0), patch_start_idx=patch_start_idx)
            dvc = (torch.sigmoid(dvc) - 0.5) * 0.2 # limit the update to [-0.1, 0.1]
            dvc = rearrange(dvc, '(b k) n h w c -> b k n h w c', b=B, k=N)

            # add blendshapes to initial canonical predictions 
            vc = vc_init[:, None] + dvc

            # Use updated canonical to pose again
            vp, _ = general_lbs(
                vc=rearrange(vc, 'b k n h w c -> (b k n) (h w) c'),
                pose=rearrange(pose_expanded, 'b k n c -> (b k n) c'),
                lbs_weights=rearrange(w_expanded, 'b k n h w j -> (b k n) (h w) j'),
                J=rearrange(joints, 'b k n j c -> (b k n) j c'),
                parents=self.smpl_model.parents 
            )
            vp = rearrange(vp, '(b k n) (h w) c -> b k n h w c', b=B, n=N, k=N, h=H, w=W)
            ret['vp'] = vp
            ret['dvc'] = dvc
            ret['vc'] = vc

        return ret 
    



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
    



        # import matplotlib.pyplot as plt
        # # Debug visualization of vp_init first pose
        # fig_debug = plt.figure(figsize=(20, 4))
        # for n in range(N):
        #     ax = fig_debug.add_subplot(1, N, n+1, projection='3d')
        #     points = vp_init[0, 0, n].cpu().detach().numpy().reshape(-1, 3)  # Flatten H,W dims
        #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
        #               c='blue', s=0.1, alpha=0.5)
        #     ax.view_init(elev=10, azim=20, vertical_axis='y')
        #     ax.set_box_aspect([1, 1, 1])
        #     ax.grid(False)
        #     ax.set_xticks([])
        #     ax.set_yticks([]) 
        #     ax.set_zticks([])
        #     ax.set_title(f'Frame {n}')
        # plt.tight_layout()
        # plt.savefig('debug_vp_init.png', dpi=300)
        # plt.close()

        # import ipdb; ipdb.set_trace()
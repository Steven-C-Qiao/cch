import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pytorch3d.structures import Pointclouds

from core.heads.dpt_head import DPTHead
from core.models.cch_aggregator import Aggregator
from core.models.sapiens_wrapper import SapiensWrapper
from core.utils.general_lbs import general_lbs
from core.configs.model_size_cfg import MODEL_CONFIGS


class CCH(nn.Module):
    def __init__(self, cfg, smpl_male, smpl_female, sapiens=None):
        """
        Given a batch of normal images, predict pixel-aligned canonical space position and uv coordinates
        """
        super(CCH, self).__init__()
        self.smpl_male = smpl_male
        self.smpl_female = smpl_female
        self.parents = smpl_male.parents
        self.cfg = cfg
        self.image_size = cfg.DATA.IMAGE_SIZE
        self.use_sapiens = cfg.MODEL.USE_SAPIENS

        # model_cfg = MODEL_CONFIGS[cfg.MODEL.SIZE]
        s1_cfg = MODEL_CONFIGS[cfg.MODEL.CANONICAL_STAGE_SIZE]
        s2_cfg = MODEL_CONFIGS[cfg.MODEL.PBS_STAGE_SIZE]

        self.model_skinning_weights = cfg.MODEL.SKINNING_WEIGHTS
        self.model_pbs = cfg.MODEL.POSE_BLENDSHAPES

        self.sapiens = sapiens
        if self.sapiens is not None:
            self.sapiens_project = nn.Conv2d(1024, s1_cfg['out_channels'][0], kernel_size=1, stride=1, padding=0)


        self.aggregator = Aggregator(
            patch_embed=s1_cfg['patch_embed'],
            img_size=self.image_size, 
            patch_size=s1_cfg['patch_size'], 
            embed_dim=s1_cfg['embed_dim'], 
            depth=s1_cfg['depth']
        )
        self.canonical_head = DPTHead(
            dim_in=2*s1_cfg['embed_dim'], 
            output_dim=4, 
            activation="inv_log", 
            conf_activation="expp1",
            out_channels=s1_cfg['out_channels'],
            features=s1_cfg['features'],
            intermediate_layer_idx=s1_cfg['intermediate_layer_idx']
        )

        if self.model_skinning_weights:
            self.skinning_head = DPTHead(
                dim_in=2*s1_cfg['embed_dim'], 
                output_dim=25, 
                activation="inv_log", 
                conf_activation="expp1", 
                additional_conditioning_dim=3,
                out_channels=s1_cfg['out_channels'],
                features=s1_cfg['features'],
                intermediate_layer_idx=s1_cfg['intermediate_layer_idx']
            )

        if self.model_pbs:
            self.pbs_aggregator = Aggregator(
                patch_embed="conv", 
                input_channels=6,
                img_size=self.image_size, 
                patch_size=s2_cfg['patch_size'], 
                embed_dim=s2_cfg['embed_dim'], 
                depth=s2_cfg['depth']
            )
            self.pbs_head = DPTHead(
                dim_in=2*s2_cfg['embed_dim'] + 2*s1_cfg['embed_dim'], 
                output_dim=4, 
                activation="inv_log", 
                conf_activation="expp1",
                out_channels=s2_cfg['out_channels'],
                features=s2_cfg['features'],
                intermediate_layer_idx=s2_cfg['intermediate_layer_idx']
            )

        # self._count_parameters()
            
    def forward(self, batch):
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
        images = batch['imgs']
        pose = batch['pose']
        joints = batch['smpl_T_joints']
        w_smpl = batch['smpl_w_maps']
        mask = batch['masks']
        


        ret = {}

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, N, C_in, H, W = images.shape

        if self.use_sapiens:
            sapiens_images = batch['sapiens_images']
            sapiens_images = rearrange(sapiens_images, 'b n c h w -> (b n) c h w')
            sapiens_features = self.sapiens(sapiens_images)
            # sapiens_features = torch.nn.functional.adaptive_avg_pool2d(sapiens_features, (16, 16))
            sapiens_features = self.sapiens_project(sapiens_features)
        else:
            sapiens_features = None


        canonical_aggregated_tokens_list, patch_start_idx = self.aggregator(images) 

        vc_init, vc_init_conf = self.canonical_head(canonical_aggregated_tokens_list, images, patch_start_idx=patch_start_idx, sapiens_features=sapiens_features)
        vc_init = torch.clamp(vc_init, -2, 2)

        vc_init = vc_init * mask.unsqueeze(-1) # Mask background pixels to origin, important for backward chamfer metrics

        ret['vc_init'] = vc_init
        ret['vc_init_conf'] = vc_init_conf


        if self.model_skinning_weights:
            w, w_conf = self.skinning_head(canonical_aggregated_tokens_list, images, patch_start_idx=patch_start_idx, additional_conditioning=vc_init, sapiens_features=sapiens_features)
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
            parents=self.parents 
        )
        vp_init = rearrange(vp_init, '(b k) (n h w) c -> b k n h w c', b=B, k=N, n=N, h=H, w=W)
        J_init = rearrange(J_init, '(b k) j c -> b k j c', b=B, k=N)

        vp_init = vp_init * (mask.unsqueeze(1).repeat(1, N, 1, 1, 1).unsqueeze(-1))

        ret['vp_init'] = vp_init
        ret['J_init'] = J_init


        if self.model_pbs:
            pbs_aggregator_input = torch.cat([rearrange(vc_init_expanded, 'b k n h w c -> (b k) n c h w'),
                                              rearrange(vp_init, 'b k n h w c -> (b k) n c h w')], dim=-3).contiguous()
            
            pbs_aggregated_tokens_list, patch_start_idx = self.pbs_aggregator(pbs_aggregator_input)


            # expand aggregated_tokens_list and concat with pbs_aggregated_tokens_list
            full_tokens_list = [torch.concat([pbs_tokens, agg_tokens.expand(pbs_tokens.shape[0], *agg_tokens.shape[1:])], dim=-1) 
                                for pbs_tokens, agg_tokens in zip(pbs_aggregated_tokens_list, canonical_aggregated_tokens_list)]
            
            dvc, dvc_conf = self.pbs_head(full_tokens_list, images.repeat_interleave(N, dim=0), patch_start_idx=patch_start_idx, sapiens_features=sapiens_features)
            dvc = (torch.sigmoid(dvc) - 0.5) * 0.2 # limit the update to [-0.1, 0.1]
            dvc = rearrange(dvc, '(b k) n h w c -> b k n h w c', b=B, k=N)
            dvc_conf = rearrange(dvc_conf, '(b k) n h w -> b k n h w', b=B, k=N)

            vc = vc_init_expanded + dvc


            vp, _ = general_lbs(
                vc=rearrange(vc, 'b k n h w c -> (b k) (n h w) c'),
                pose=rearrange(pose, 'b k c -> (b k) c'),
                lbs_weights=rearrange(w_expanded, 'b k n h w j -> (b k) (n h w) j'),
                J=rearrange(joints, 'b k j c -> (b k) j c'),
                parents=self.parents 
            )
            vp = rearrange(vp, '(b k) (n h w) c -> b k n h w c', b=B, n=N, k=N, h=H, w=W)

            vp = vp * (mask.unsqueeze(1).repeat(1, N, 1, 1, 1).unsqueeze(-1))
            
            ret['vp'] = vp
            ret['dvc'] = dvc
            ret['dvc_conf'] = dvc_conf
            ret['vc'] = vc


        return ret 
    



    def _count_parameters(self):
        """
        Count number of trainable parameters in all model components
        """
        param_counts = {}
        
        # Count aggregator parameters
        if hasattr(self, 'aggregator'):
            param_counts['aggregator'] = sum(p.numel() for p in self.aggregator.parameters() if p.requires_grad)
            print(f'Aggregator parameters: {param_counts["aggregator"]:,}')
            
        # Count canonical head parameters
        if hasattr(self, 'canonical_head'):
            param_counts['canonical_head'] = sum(p.numel() for p in self.canonical_head.parameters() if p.requires_grad)
            print(f'Canonical Head parameters: {param_counts["canonical_head"]:,}')
            
        # Count skinning head parameters if enabled
        if self.model_skinning_weights and hasattr(self, 'skinning_head'):
            param_counts['skinning_head'] = sum(p.numel() for p in self.skinning_head.parameters() if p.requires_grad)
            print(f'Skinning Head parameters: {param_counts["skinning_head"]:,}')
            
        # Count PBS components if they exist
        if self.model_pbs:
            pbs_agg_params = sum(p.numel() for p in self.pbs_aggregator.parameters() if p.requires_grad)
            pbs_head_params = sum(p.numel() for p in self.pbs_head.parameters() if p.requires_grad)
            param_counts['pbs_aggregator'] = pbs_agg_params
            param_counts['pbs_head'] = pbs_head_params
            print(f'PBS Aggregator parameters: {pbs_agg_params:,}')
            print(f'PBS Head parameters: {pbs_head_params:,}')
            
        # Calculate total
        total_params = sum(param_counts.values())
        param_counts['total'] = total_params
        print(f'Total parameters: {total_params:,}')
        
        return param_counts
    



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
import os
import torch
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.loss import chamfer_distance

from core.configs import paths 

from core.models.smpl import SMPL
from core.models.cch import CCH 

from core.losses.cch_loss import CCHLoss

from core.utils.renderer import SurfaceNormalRenderer 
from core.utils.sample_utils import sample_cameras
from core.utils.general_lbs import general_lbs
from core.utils.visualiser import Visualiser
from core.utils.simple_feature_renderer import FeatureRenderer

class CCHTrainer(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 dev=False, 
                 vis_save_dir=None):
        
        super().__init__()
        self.save_scenepic = True 
        self.dev = dev
        self.cfg = cfg
        self.normalise = cfg.DATA.NORMALISE
        self.vis_frequency = cfg.VISUALISE_FREQUENCY if not dev else 5
        self.image_size = cfg.DATA.IMG_SIZE
 
        self.renderer = SurfaceNormalRenderer(image_size=(224, 224))

        self.feature_renderer = FeatureRenderer(image_size=(224, 224))

        self.smpl_model = SMPL(
            model_path=paths.SMPL,
            num_betas=10,
            gender=cfg.MODEL.GENDER
        )
        # smpl_faces = torch.tensor(smpl_model.faces, dtype=torch.int32)
        # self.register_buffer('smpl_faces', smpl_faces)
        for param in self.smpl_model.parameters():
            param.requires_grad = False

        self.model = CCH(
            cfg=cfg,
            smpl_model=self.smpl_model
        )

        # Freeze aggregator and canonical_head
        for param in self.model.aggregator.parameters():
            param.requires_grad = False
        for param in self.model.canonical_head.parameters():
            param.requires_grad = False

        self.criterion = CCHLoss(cfg)
        self.visualiser = Visualiser(save_dir=vis_save_dir)

        self.save_hyperparameters(ignore=['smpl_model'])
        

    def forward(self, batch, pose=None, joints=None, w_smpl=None, mask=None):
        return self.model(batch, pose=pose, joints=joints, w_smpl=w_smpl, mask=mask)
    
    def on_train_epoch_start(self):
        self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train'):
        if not self.dev:
            batch = self.process_inputs(batch, batch_idx, normalise=self.normalise)
        if self.global_step == 0:
            if self.dev:
                batch = self.process_inputs(batch, batch_idx, normalise=self.normalise)
            self.first_batch = batch
            # self.visualiser.visualise_input_normal_imgs(batch['normal_imgs'])
        if self.dev:    
            batch = self.first_batch

        B, N = batch['pose'].shape[:2]
        vp = batch['v_posed']
        vp_sampled = batch['v_posed_samples']
        masks = batch['masks']
        pose = batch['pose']
        joints = batch['joints']
        normal_maps = batch['normal_imgs']
        w_smpl = batch['w_pm']
        vc = batch['vc_pm']
        dvc_pm_target = batch['dvc_pm']
        vertex_visibility = batch['vertex_visibility']


        # ------------------- forward pass -------------------
        preds = self(
            normal_maps, 
            pose=pose, 
            joints=joints, 
            w_smpl=w_smpl, 
            mask=masks
        )
        vc_init_pred, vc_pred,  vc_conf = preds['vc_init_pred'], preds['vc_pred'], preds['vc_conf']
        vp_init_pred, vp_pred = preds['vp_init_pred'], preds['vp_pred']

        if self.cfg.MODEL.SKINNING_WEIGHTS:
            w_pred, w_conf = preds['w_pred'], preds['w_conf_pred']
        else:
            w_pred, w_conf = w_smpl, None

        if self.cfg.MODEL.POSE_BLENDSHAPES:
            dvc_pred, dvc_conf = preds['dvc_pred'], None
        else:
            dvc_pred, dvc_conf = None, None



        loss, loss_dict = self.criterion(
            vp=vp_sampled,
            vp_pred=vp_pred,
            vc=vc,
            # vc_pred=vc_init_pred, 
            conf=vc_init_pred_conf,
            mask=masks,
            w_pred=w_pred,
            w_smpl=w_smpl,
            dvc_pred=dvc_pred,
            dvc_conf=dvc_conf,
            dvc_pm_target=dvc_pm_target,
            epoch=self.current_epoch
        )
        self.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)


        self.metrics(
            vc=rearrange(vc, 'b n h w c -> (b n) (h w) c'), 
            vc_pred=rearrange(vc_init_pred, 'b n h w c -> (b n) (h w) c'), 
            vp=rearrange(vp, 'b k v c -> (b k) v c'), 
            vp_pred=rearrange(vp_pred, 'b k n h w c -> (b k) (n h w) c'), 
            conf=rearrange(vc_init_pred_conf, 'b n h w -> (b n) (h w)'), 
            mask=rearrange(masks, 'b n h w -> (b n) (h w)'), 
            split=split, 
            B=B,
            N=N
        )
        

        # Visualise
        # if (self.global_step % self.vis_frequency == 0 and self.global_step > 0) or (self.global_step == 1):
        self.visualiser.visualise(
            normal_maps=normal_maps.cpu().detach().numpy(),
            vp=vp.cpu().detach().numpy(),
            vc=vc.cpu().detach().numpy(),
            vp_pred=vp_pred.cpu().detach().numpy(),
            vp_init_pred=vp_init_pred.cpu().detach().numpy(),
            vc_pred=vc_pred.cpu().detach().numpy(),
            vc_init_pred=vc_init_pred.cpu().detach().numpy(),
            conf=vc_init_pred_conf.cpu().detach().numpy(),
            mask=masks.cpu().detach().numpy(),
            vertex_visibility=vertex_visibility.cpu().detach().numpy(),
            color=np.argmax(w_pred.cpu().detach().numpy(), axis=-1),
            dvc=dvc_pm_target.cpu().detach().numpy(),
            dvc_pred=dvc_pred.cpu().detach().numpy(),
            no_annotations=True,
            plot_error_heatmap=True
        )

        # import ipdb 
        # ipdb.set_trace()

        # For app.animate_pm.py 
        # self.save_avatar(vc_pred, vc_conf, w_pred, joints, masks, w_smpl=w_smpl)

        return loss 
    

    def metrics(self, vc=None, vc_pred=None, vp=None, vp_pred=None,
                conf=None, mask=None, vp_mask=None, split='train', B=None, N=None):
        if conf is not None:
            conf_threshold = 0.08
            conf_mask = (1/conf) < conf_threshold
            full_mask = (mask * conf_mask).bool()
        else:
            full_mask = mask.bool()

        # ----------------------- vc -----------------------
        vc_pm_dist = torch.norm(vc - vc_pred, dim=-1) * full_mask 
        vc_pm_dist = vc_pm_dist.sum() / (full_mask.sum() + 1e-6) * 100.0


        # ----------------------- vp -----------------------
        full_vp_mask = rearrange(full_mask, '(b n) (h w) -> b n h w', 
                                 b=B, n=N, h=self.image_size, w=self.image_size)
        full_vp_mask = full_vp_mask[:, None].repeat(1, N, 1, 1, 1)
        full_vp_mask = rearrange(full_vp_mask, 'b k n h w -> (b k) (n h w)')

        vp_dist_squared, _ = chamfer_distance(vp_pred, vp, batch_reduction=None, point_reduction=None)
        vpp2vp_dist = torch.sqrt(vp_dist_squared[0])
        masked_vpp2vp_dist = vpp2vp_dist * full_vp_mask
        vpp2vp_dist = masked_vpp2vp_dist.sum() / (full_vp_mask.sum() + 1e-6) * 100.0

        vp2vpp_dist = torch.sqrt(vp_dist_squared[1])
        vp2vpp_dist = vp2vpp_dist.mean() * 100.0

        show_prog_bar = (split == 'train')

        self.log(f'{split}_vc_pm_dist', vc_pm_dist,  on_step=True, on_epoch=True, prog_bar=show_prog_bar, sync_dist=True, rank_zero_only=True)
        self.log(f'{split}_vpp2vp_cfd', vpp2vp_dist, on_step=True, on_epoch=True, prog_bar=show_prog_bar, sync_dist=True, rank_zero_only=True)
        self.log(f'{split}_vp2vpp_cfd', vp2vpp_dist, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)

        return {
            'vc_pm_dist': vc_pm_dist,
            'vpp2vp_cfd': vpp2vp_dist,
            'vp2vpp_cfd': vp2vpp_dist,
        }



    def process_inputs(self, batch, batch_idx, normalise=False):
        """
        Batch of size B contains N frames of the same person. For each frame, sample a random camera pose and render.
        """
        with torch.no_grad():
            t = batch['transl'] # B, N, 3
            vp = batch['v_posed'] # B, N, 6890, 3

            B, N, V, _ = vp.shape

            vp = vp - t[:, :, None, :]
            batch['v_posed'] = vp


            if normalise:
                vc = batch['first_frame_v_cano'] # B, 6890, 3
                subject_height = (vc[..., 1].max(dim=-1).values - vc[..., 1].min(dim=-1).values)

                batch['first_frame_v_cano'] = vc / subject_height[:, None, None] * 1.7 # B, 6890, 3
                batch['v_posed'] = batch['v_posed'] / subject_height[:, None, None, None] * 1.7 # B, N, 6890, 3
                batch['v_cano'] = batch['v_cano'] / subject_height[:, None, None, None] * 1.7 # B, N, 6890, 3
                

            smpl_output = self.smpl_model(
                betas=batch['betas'],
                body_pose = torch.zeros((B, 69)).to(self.device),
                global_orient = torch.zeros((B, 3)).to(self.device)
            )
            joints = smpl_output.joints[:, :24]

            # CAPE provided naked shape is bad, 
            # add a correction to the joints by scaling height of the provided naked shape to first_frame_v_cano
            naked_height = (smpl_output.vertices[:, :, 1].max(dim=-1).values - smpl_output.vertices[:, :, 1].min(dim=-1).values)
            joints = joints * (subject_height[:, None, None] / naked_height[:, None, None]) 

            if normalise:
                joints = joints / subject_height[:, None, None] * 1.7

            w = (self.smpl_model.lbs_weights)[None, None].repeat(B, N, 1, 1)


            R, T = sample_cameras(B, N, self.cfg.DATA)
            R = R.to(self.device)
            T = T.to(self.device)

            # self.feature_renderer._set_cameras(R, T)
            # render normal images
            ret = self.renderer(
                vertices=batch['v_posed'], 
                R=R, 
                T=T, 
                skinning_weights=w,
                first_frame_v_cano=batch['first_frame_v_cano'],
            )
            normal_imgs = ret['normals'].permute(0, 1, 4, 2, 3)
            masks = ret['masks'].permute(0, 1, 4, 2, 3).squeeze()

            # ------------------- dvc maps -------------------
            dvc = batch['v_cano'] - batch['first_frame_v_cano'][:, None]
            verts_in = batch['v_posed'][:, None].repeat(1, N, 1, 1, 1)
            dvc_in = dvc[:, :, None].repeat(1, 1, N, 1, 1)
            R_in = R[:, None].repeat(1, N, 1, 1, 1)
            T_in = T[:, None].repeat(1, N, 1, 1)    
            dvc_maps = self.renderer(
                vertices=rearrange(verts_in, 'b k n v c -> (b k) n v c'), 
                R=rearrange(R_in, 'b k n x y -> (b k) n x y'), 
                T=rearrange(T_in, 'b k n x -> (b k) n x'), 
                temp = rearrange(dvc_in, 'b k n v c -> (b k n) v c')
            )
            dvc_maps = rearrange(dvc_maps, '(b k) n h w c -> b k n h w c', b=B, k=N)
            
            batch['normal_imgs'] = normal_imgs
            batch['R'] = R
            batch['T'] = T
            batch['joints'] = joints
            batch['masks'] = masks
            batch['w_pm'] = ret['skinning_weights_maps']
            batch['vc_pm'] = ret['canonical_color_maps']
            batch['vertex_visibility'] = ret['vertex_visibility']
            batch['dvc_pm'] = dvc_maps
        return batch 
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='val')
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    

    

    def configure_optimizers(self):
        # Only include trainable parameters (excluding frozen aggregator and canonical_head)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        params = trainable_params + list(self.criterion.parameters())
        optimizer = optim.Adam(params, lr=self.cfg.TRAIN.LR)
        
        # Add cosine learning rate scheduler
        if self.cfg.TRAIN.LR_SCHEDULER == 'cosine': 
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.TRAIN.NUM_EPOCHS,  # Total number of epochs
                eta_min=self.cfg.TRAIN.LR * 0.1  # Minimum learning rate (1% of initial LR)
            )
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer


    # dvc_maps = []
            # for k in range(N): # For each pose, render dvc maps for all views
            #     vertices = batch['v_posed'][:, [k]]
            #     dvc_maps.append(self.feature_renderer(
            #         vertices=vertices,
            #         dvc=dvc[:, [k]]
            #     )['dvc_maps'])
            # dvc_maps = torch.stack(dvc_maps, dim=1)


            # """ b k n h w c """
            # b, k, n = 0, 2, 0
            # import matplotlib.pyplot as plt
            # dvc_maps[~masks[:, None].repeat(1, N, 1,1,1,1).bool().squeeze()] = 0
            # plt.figure(figsize=(10, 10))

            # for k in range(N):
            #     for n in range(N):
            #         plt.subplot(N, N, k*N + n + 1)
            #         plt.imshow(torch.norm(dvc_maps[b, k, n], dim=-1).cpu().detach().numpy())
            #         plt.title(f'{k}th pose, on the {n}th pm')
            #         plt.axis('off')
            # # plt.colorbar()
            # plt.savefig(f'dvc_maps.png')
            # plt.show()

            # import ipdb 
            # ipdb.set_trace()



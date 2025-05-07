import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from loguru import logger
from einops import rearrange


from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

from core.configs import paths 
from core.losses.cch_loss import CCHLoss

from core.models.smpl import SMPL
from core.models.cch import CCH 

from core.utils.normal_renderer import SurfaceNormalRenderer 
from core.utils.sample_utils import sample_cameras
from core.utils.general_lbs import general_lbs




# from losses.deterministic_loss import DeterministicLoss
# from losses.homosced_loss import HomoscedWeightedLoss
# from utils.vis_utils import SculpterVisualiser
# from utils.label_conversions import convert_joints2D_to_hmaps
# from utils.geometry_utils import compute_smpl, compute_smpl_from_canonical_mesh
# from utils.augmentation.proxy_rep import random_swap_joints2D, random_joints2D_deviation, random_remove_joints2D

# from metrics.metrics_calculator import MetricsCalculator

class CCHTrainer(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 dev=False, 
                 vis_save_dir=None):
        
        super().__init__()
        self.dev = dev
        self.cfg = cfg
        self.normalise = False
        self.vis_frequency = cfg.VISUALISE_FREQUENCY if not dev else 5
 
        self.renderer = SurfaceNormalRenderer(image_size=(224, 224))

        self.smpl_model = SMPL(
            model_path=paths.SMPL,
            num_betas=cfg.MODEL.NUM_SMPL_BETAS,
            gender=cfg.MODEL.GENDER
        )
        # smpl_faces = torch.tensor(smpl_model.faces, dtype=torch.int32)
        # self.register_buffer('smpl_faces', smpl_faces)
        # for param in smpl_model.parameters():
        #     param.requires_grad = False

        self.model = CCH(
            cfg=cfg,
            smpl_model=self.smpl_model
        )

        self.criterion = CCHLoss()
        
        
        # visualiser = Visualiser(save_dir=vis_save_dir)
        
        
        # self.model = model
        # self.smpl_model = smpl_model
        # self.visualiser = visualiser
        # # self.criterion = DeterministicLoss(loss_cfg=cfg.LOSS, backbone_losses=cfg.FINETUNE_BACKBONE)
        # self.criterion = HomoscedWeightedLoss(loss_cfg=cfg.LOSS, backbone_losses=cfg.TRAIN_BACKBONE)
        # self.metrics_calculator = MetricsCalculator()

        self.save_hyperparameters(ignore=['smpl_model', 'edge_detect_model'])
        

    def forward(self, batch):
        return self.model(batch)
    
    def on_train_epoch_start(self):
        pass 
        # self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train'):
        batch = self.process_inputs(batch, batch_idx)
        B, N = batch['pose'].shape[:2]
        vp = batch['v_posed']
        
        if batch_idx == 0:
            self.first_batch = batch
        if self.dev:    
            batch = self.first_batch

        global_pose, body_pose = batch['pose'][..., :3], batch['pose'][..., 3:]
        joints = batch['joints']
        normal_images = batch['normal_imgs']
            
        preds = self(normal_images)

        # ----------------- Loss -----------------
        pred_vc, pred_w = preds['vc'], preds['w']


        parents = self.smpl_model.parents
        
        vp_pred, joints_pred = general_lbs(
            vc=rearrange(pred_vc, 'b n h w c -> (b n) (h w) c'),
            pose=rearrange(batch['pose'], 'b n c -> (b n) c'),
            lbs_weights=rearrange(pred_w, 'b n h w c -> (b n) (h w) c'),
            J=joints.repeat_interleave(batch['pose'].shape[1], dim=0),
            parents=parents#parents[None].repeat(B * N, 1)
        )

        loss, loss_normals = self.criterion(vp_pred, rearrange(vp, 'b n v c -> (b n) v c'))

        # # Visualise and log
        # if batch_idx % self.vis_frequency == 0:

            # self.visualiser.visualise(normal_images, predicted_normal_images)
            # self.logger.experiment.add_figure(f'{split}_pred', self.visualiser.fig, self.global_step)

        # self.metrics_calculator.update(pred_dict, targets_dict, self.cfg.TRAIN.BATCH_SIZE)

        # d_vc = torch.cat(pred_dict['pred_d_vc_list'], dim=0) 
        # self.log(f'{split}_d_vc_mean_magnitude', torch.mean(torch.abs(d_vc)), on_step=True)
        # self.log(f'{split}_d_vc_max_magnitude', torch.max(torch.abs(d_vc)), on_step=True)
        
        # self.log(f'{split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # for k, v in loss_dict.items():
        #     self.log(f'{split}_{k}', v, on_step=True, on_epoch=True, sync_dist=True)
        
        # for metrics in self.metrics_calculator.metrics:
        #     self.log(f'{split}_{metrics}', self.metrics_calculator.metrics_dict[metrics][-1], on_step=True, on_epoch=True, sync_dist=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('loss_normals', loss_normals, on_step=True, on_epoch=True, sync_dist=True)

        return loss 
    


    def process_inputs(self, batch, batch_idx):
        """
        Batch of size B contains N frames of the same person. For each frame, sample a random camera pose and render.
        """
        with torch.no_grad():
            t = batch['transl'] # B, N, 3
            vp = batch['v_posed'] # B, N, 6890, 3

            batch_size, num_frames = t.shape[:2]


            smpl_output = self.smpl_model(
                betas=batch['betas'],
                body_pose = torch.zeros((batch_size, 69)).to(self.device),
                global_orient = torch.zeros((batch_size, 3)).to(self.device)
            )
            joints = smpl_output.joints[:, :24]

            
            R, T = sample_cameras(batch_size, num_frames, t)
            R = R.to(self.device)
            T = T.to(self.device)

            # render normal images
            normal_imgs = self.renderer(vp, R, T)
            normal_imgs = torch.tensor(normal_imgs, 
                                       dtype=torch.float32).permute(0, 1, 4, 2, 3).to(self.device)
            
            batch['normal_imgs'] = normal_imgs
            batch['R'] = R
            batch['T'] = T
            batch['joints'] = joints

        return batch 
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='val')
            # Add explicit validation logging
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        names = []
        vals = []
        for name, param in self.criterion.named_parameters():
            names.append(name.split('.')[-1])
            vals.append(torch.exp(-param.data.clone()).item())
        # logger.info(f'Current homosced weights: {list(zip(names, vals))}')


    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        optimizer = optim.Adam(params, lr=self.cfg.TRAIN.LR)
        return optimizer 
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)
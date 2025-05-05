import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from loguru import logger

from smplx import SMPL
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform

from core.configs import paths 
from core.models.cch import CCH 

from core.utils.pytorch3d_surface_normal_renderer import SurfaceNormalRenderer 
from core.utils.sample_utils import sample_cameras





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
 
        self.renderer = SurfaceNormalRenderer()

        smpl_model = SMPL(
            model_path=paths.SMPL,
            num_betas=cfg.MODEL.NUM_SMPL_BETAS,
            gender=cfg.MODEL.GENDER
        )
        smpl_faces = torch.tensor(smpl_model.faces, dtype=torch.int32)
        self.register_buffer('smpl_faces', smpl_faces)
        # for param in smpl_model.parameters():
        #     param.requires_grad = False

        self.model = CCH(
            cfg=cfg,
            smpl_model=smpl_model
        )

        
        
        # visualiser = Visualiser(save_dir=vis_save_dir)
        
        
        # self.model = model
        # self.smpl_model = smpl_model
        # self.visualiser = visualiser
        self.criterion = nn.MSELoss()
        # # self.criterion = DeterministicLoss(loss_cfg=cfg.LOSS, backbone_losses=cfg.FINETUNE_BACKBONE)
        # self.criterion = HomoscedWeightedLoss(loss_cfg=cfg.LOSS, backbone_losses=cfg.TRAIN_BACKBONE)
        # self.metrics_calculator = MetricsCalculator()

        self.save_hyperparameters(ignore=['smpl_model', 'edge_detect_model'])
        

    def forward(self, batch, iters=2, epoch=0):
        return self.model(batch, iters=iters)#, epoch=epoch)
    
    def on_train_epoch_start(self):
        self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train'):
        if batch_idx == 0:
            inputs_dict, targets_dict, num_views = self.process_inputs(batch, batch_idx)
            self.first_batch = batch
            self.first_vc = targets_dict['vc']
        if self.dev:    
            batch = self.first_batch
            
        inputs_dict, targets_dict, num_views = self.process_inputs(batch, batch_idx)

        inputs_dict['gt_pose'] = batch['pose_rotmats'].view(-1, 23, 3, 3)
        inputs_dict['gt_glob_rotmats'] = batch['glob_rotmats'].view(-1, 3, 3)
        inputs_dict['gt_cam_t'] = batch['cam_t'].view(-1, 3)

 
        pred_dict = self(inputs_dict, epoch=self.current_epoch)

        import ipdb; ipdb.set_trace()


        # loss, loss_dict = self.criterion(pred_dict['sculpted_pred_dict'], 
        #                                  targets_dict, 
        #                                  self.cfg.DATA.IMG_SIZE)

        # # Visualise and log
        # if batch_idx % self.vis_frequency == 0:
        #     # self.visualiser.visualise(pred_dict.copy(), 
        #     #                           targets_dict.copy(), 
        #     #                           batch_idx, self.current_epoch)
        #     self.visualiser.visualise_on_image(inputs_dict.copy(), 
        #                                        targets_dict.copy(), 
        #                                        pred_dict.copy(), 
        #                                        batch_idx, self.current_epoch)
        #     self.logger.experiment.add_figure(f'{split}_pred', self.visualiser.fig, self.global_step)

        # self.metrics_calculator.update(pred_dict, targets_dict, self.cfg.TRAIN.BATCH_SIZE)

        # d_vc = torch.cat(pred_dict['pred_d_vc_list'], dim=0) 
        # self.log(f'{split}_d_vc_mean_magnitude', torch.mean(torch.abs(d_vc)), on_step=True)
        # self.log(f'{split}_d_vc_max_magnitude', torch.max(torch.abs(d_vc)), on_step=True)
        
        # self.log(f'{split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # for k, v in loss_dict.items():
        #     self.log(f'{split}_{k}', v, on_step=True, on_epoch=True, sync_dist=True)
        
        # for metrics in self.metrics_calculator.metrics:
        #     self.log(f'{split}_{metrics}', self.metrics_calculator.metrics_dict[metrics][-1], on_step=True, on_epoch=True, sync_dist=True)

        return None
    


    def process_inputs(self, batch, batch_idx):
        with torch.no_grad():
            t = batch['transl']
            vp = batch['v_posed']
            p = batch['pose']
            vc = batch['v_cano']

            # num_views = t.shape[0]
            num_views = 4

            
            
            R, T = sample_cameras(vp.shape[0], num_views, t)

            # render normal images
            normal_imgs = self.renderer(vp, self.smpl_faces, R, T)

            ret = {
                'normal_imgs': normal_imgs,
                'pose': p
            }

            

        return ret 
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='val')
        return loss
    
    def on_validation_epoch_end(self):
        names = []
        vals = []
        for name, param in self.criterion.named_parameters():
            names.append(name.split('.')[-1])
            vals.append(torch.exp(-param.data.clone()).item())
        logger.info(f'Current homosced weights: {list(zip(names, vals))}')


    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.criterion.parameters())
        optimizer = optim.Adam(params, lr=self.cfg.TRAIN.LR)
        return optimizer 
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)
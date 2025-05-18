import torch
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl

from einops import rearrange

from core.configs import paths 

from core.models.smpl import SMPL
from core.models.cch import CCH 

from core.losses.cch_loss import CCHLoss

from core.utils.renderer import SurfaceNormalRenderer 
from core.utils.sample_utils import sample_cameras
from core.utils.general_lbs import general_lbs
from core.utils.visualiser import Visualiser

# from metrics.metrics_calculator import MetricsCalculator

class CCHTrainer(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 dev=False, 
                 vis_save_dir=None):
        
        super().__init__()
        self.dev = dev
        self.cfg = cfg
        self.vis_frequency = cfg.VISUALISE_FREQUENCY if not dev else 1000
 
        self.renderer = SurfaceNormalRenderer(image_size=(224, 224))

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

        self.criterion = CCHLoss(cfg)
        self.visualiser = Visualiser(save_dir=vis_save_dir)
        # self.metrics_calculator = MetricsCalculator()

        self.save_hyperparameters(ignore=['smpl_model'])
        

    def forward(self, batch):
        return self.model(batch)
    
    def on_train_epoch_start(self):
        self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train'):
        if not self.dev: # more randomness in process_inputs, naive set first batch doesn't work 
            batch = self.process_inputs(batch, batch_idx)
        if batch_idx == 0:
            if self.dev:
                batch = self.process_inputs(batch, batch_idx)
            self.first_batch = batch
            self.visualiser.visualise_input_normal_imgs(batch['normal_imgs'])
        if self.dev:    
            batch = self.first_batch

        B, N = batch['pose'].shape[:2]
        vp = batch['v_posed']
        mask = batch['masks']
        # global_pose, body_pose = batch['pose'][..., :3], batch['pose'][..., 3:]
        joints = batch['joints']
        normal_images = batch['normal_imgs']
        w_smpl = batch['skinning_weights_maps']
        vc_gt = batch['canonical_color_maps']


        # ------------------- forward pass -------------------
        preds = self(normal_images)

        vc_pred, dw_pred = preds['vc'], preds['w']


        # w_pred = dw_pred + coarse_skinning_weights_maps
        w_pred = w_smpl


        vp_pred, joints_pred = general_lbs(
            vc=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'),
            pose=rearrange(batch['pose'], 'b n c -> (b n) c'),
            lbs_weights=rearrange(w_pred, 'b n h w c -> (b n) (h w) c'),
            J=joints.repeat_interleave(batch['pose'].shape[1], dim=0),
            parents=self.smpl_model.parents 
        )
        # joints_pred = rearrange(joints_pred, '(b n) j c -> b n j c', b=B, n=N)


        loss, loss_dict = self.criterion(vp=rearrange(vp, 'b n v c -> (b n) v c'),
                                         vp_pred=vp_pred,
                                         vc=rearrange(vc_gt, 'b n h w c -> (b n) (h w) c'),
                                         vc_pred=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'),
                                         mask=mask,
                                         )
        

        # Visualise and log
        vp_pred = rearrange(vp_pred, '(b n) v c -> b n v c', b=B, n=N)
        vc_pred = rearrange(vc_pred, 'b n h w c -> b n (h w) c', b=B, n=N)
        if self.global_step % self.vis_frequency == 0:
            w_argmax = np.argmax(w_pred.cpu().detach().numpy(), axis=-1)
            self.visualiser.visualise_vp(vp.cpu().detach().numpy(), 
                                         vp_pred.cpu().detach().numpy(), 
                                         mask.cpu().detach().numpy(),
                                         color=w_argmax)
            self.visualiser.visualise_vc(vc_pred.cpu().detach().numpy(), 
                                         mask.cpu().detach().numpy(),
                                         color=w_argmax)
            # self.logger.experiment.add_figure(f'{split}_pred', self.visualiser.fig, self.global_step)
            self.visualiser.visualise_vc_as_image(vc_pred.cpu().detach().numpy(), 
                                                  vc_gt.cpu().detach().numpy(),
                                                  mask.cpu().detach().numpy())
            

        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f'{split}_{k}', v, on_step=True, on_epoch=True, sync_dist=True)
        
        # self.metrics_calculator.update(pred_dict, targets_dict, self.cfg.TRAIN.BATCH_SIZE)
        # for metrics in self.metrics_calculator.metrics:
        #     self.log(f'{split}_{metrics}', self.metrics_calculator.metrics_dict[metrics][-1], on_step=True, on_epoch=True, sync_dist=True)


        if batch_idx % 200 == 0 and batch_idx > 0:
            # import ipdb; ipdb.set_trace()
            pass 

        return loss 
    


    def process_inputs(self, batch, batch_idx):
        """
        Batch of size B contains N frames of the same person. For each frame, sample a random camera pose and render.
        """
        with torch.no_grad():
            t = batch['transl'] # B, N, 3
            vp = batch['v_posed'] # B, N, 6890, 3

            vp = vp - t[:, :, None, :]
            batch['v_posed'] = vp

            batch_size, num_frames = t.shape[:2]


            smpl_output = self.smpl_model(
                betas=batch['betas'],
                body_pose = torch.zeros((batch_size, 69)).to(self.device),
                global_orient = torch.zeros((batch_size, 3)).to(self.device)
            )
            joints = smpl_output.joints[:, :24]

            w = (self.smpl_model.lbs_weights)[None, None].repeat(batch_size, num_frames, 1, 1)

            
            # R, T = sample_cameras(batch_size, num_frames, t)
            R, T = sample_cameras(batch_size, num_frames)
            R = R.to(self.device)
            T = T.to(self.device)

            # render normal images
            ret = self.renderer(vp, 
                                R, 
                                T, 
                                skinning_weights=w,
                                first_frame_v_cano=batch['first_frame_v_cano'])
            normal_imgs = torch.tensor(ret['normals'], 
                                       dtype=torch.float32).permute(0, 1, 4, 2, 3).to(self.device)
            
            masks = torch.tensor(ret['masks'], 
                                dtype=torch.float32).permute(0, 1, 4, 2, 3).to(self.device)
            
            skinning_weights_maps = torch.tensor(ret['skinning_weights_maps'], 
                                                dtype=torch.float32).to(self.device)
            
            canonical_color_maps = torch.tensor(ret['canonical_color_maps'], 
                                                dtype=torch.float32).to(self.device)
            
            
            batch['normal_imgs'] = normal_imgs
            batch['R'] = R
            batch['T'] = T
            batch['joints'] = joints
            batch['masks'] = masks
            batch['skinning_weights_maps'] = skinning_weights_maps
            batch['canonical_color_maps'] = canonical_color_maps
        return batch 
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='val')
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
        
        
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             grad_norm = param.grad.norm().item()
    #             if grad_norm > 10:
    #                 print(f"Large gradient in {name}: {grad_norm}")
    
import os
import torch
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange


import scenepic as sp 
from matplotlib import cm, pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import viridis


from core.configs import paths 

from core.models.smpl import SMPL
from core.models.cch import CCH 

from core.losses.cch_loss import CCHLoss

from core.utils.renderer import SurfaceNormalRenderer 
from core.utils.sample_utils import sample_cameras
from core.utils.general_lbs import general_lbs
from core.utils.visualiser import Visualiser

class CCHTrainer(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 dev=False, 
                 vis_save_dir=None):
        
        super().__init__()
        self.dev = dev
        self.cfg = cfg
        self.normalise = cfg.DATA.NORMALISE
        self.vis_frequency = cfg.VISUALISE_FREQUENCY if not dev else 100
 
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

        self.save_hyperparameters(ignore=['smpl_model'])
        

    def forward(self, batch):
        return self.model(batch)
    
    def on_train_epoch_start(self):
        self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train'):
        try:
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
            mask = batch['masks'].squeeze()
            # global_pose, body_pose = batch['pose'][..., :3], batch['pose'][..., 3:]
            joints = batch['joints']
            normal_maps = batch['normal_imgs']
            w_smpl = batch['skinning_weights_maps']
            vc = batch['canonical_color_maps']


            # ------------------- forward pass -------------------
            preds = self(normal_maps)
            vc_pred, vc_conf = preds['vc'], preds['vc_conf']


            if self.cfg.MODEL.SKINNING_WEIGHTS:
                w_pred, w_conf = preds['w'], preds['w_conf']
                # w_pred = w_smpl + dw_pred
            else:
                w_pred = w_smpl
                w_conf = None


            vp_pred, joints_pred = general_lbs(
                vc=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'),
                pose=rearrange(batch['pose'], 'b n c -> (b n) c'),
                lbs_weights=rearrange(w_pred, 'b n h w j -> (b n) (h w) j'),
                J=joints.repeat_interleave(batch['pose'].shape[1], dim=0),
                parents=self.smpl_model.parents 
            )
            vp_pred = rearrange(vp_pred, '(b n) v c -> b n v c', b=B, n=N)
            # joints_pred = rearrange(joints_pred, '(b n) j c -> b n j c', b=B, n=N)


            loss, loss_dict = self.criterion(vp=batch['sampled_posed_points'],
                                             vp_pred=vp_pred,
                                             vc=vc,
                                             vc_pred=vc_pred, 
                                             conf=vc_conf,
                                             mask=mask,
                                             w_pred=w_pred,
                                             w_smpl=w_smpl)

            # Visualise and log
            if self.global_step % self.vis_frequency == 0:
                self.visualiser.visualise(normal_maps=normal_maps.cpu().detach().numpy(),
                                          vp=vp.cpu().detach().numpy(),
                                          vc=vc.cpu().detach().numpy(),
                                          vp_pred=vp_pred.cpu().detach().numpy(),
                                          vc_pred=vc_pred.cpu().detach().numpy(),
                                          conf=vc_conf.cpu().detach().numpy(),
                                          mask=mask.cpu().detach().numpy(),
                                          vertex_visibility=batch['vertex_visibility'].cpu().detach().numpy(),
                                          color=np.argmax(w_pred.cpu().detach().numpy(), axis=-1),
                                          no_annotations=True,
                                          plot_error_heatmap=True)

            self.log(f'{split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
            for k, v in loss_dict.items():
                if k != 'total_loss':
                    self.log(f'{split}_{k}', v, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)

            vc_avg_err = torch.linalg.norm(vc - vc_pred, dim=-1) * mask 
            vc_avg_err = vc_avg_err.sum() / mask.sum()
            self.log(f'{split}_vc_avg_dist', vc_avg_err, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)

            # if w_pred is not None:
            #     self.log(f'{split}_wpred_max', w_pred.max(), on_step=True, on_epoch=True, sync_dist=True)

            if batch_idx % 10 == 0 and batch_idx > 0:
                # import ipdb; ipdb.set_trace()
                pass

            # import ipdb; ipdb.set_trace()
            # print(loss_dict)

            return loss 
        except RuntimeError as e:
            if "NCCL" in str(e):
                print(f"NCCL error on GPU {self.global_rank}: {str(e)}")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Retry the step
                return self.training_step(batch, batch_idx, split)
            raise
    


    def process_inputs(self, batch, batch_idx, normalise=False):
        """
        Batch of size B contains N frames of the same person. For each frame, sample a random camera pose and render.
        """
        with torch.no_grad():
            t = batch['transl'] # B, N, 3
            vp = batch['v_posed'] # B, N, 6890, 3

            vp = vp - t[:, :, None, :]
            batch['v_posed'] = vp

            batch_size, num_frames = t.shape[:2]

            if normalise:
                vc = batch['first_frame_v_cano'] # B, 6890, 3
                subject_height = (vc[..., 1].max(dim=-1).values - vc[..., 1].min(dim=-1).values)

                batch['first_frame_v_cano'] = vc / subject_height[:, None, None] # B, 6890, 3
                batch['v_posed'] = batch['v_posed'] / subject_height[:, None, None, None] # B, N, 6890, 3
                


            smpl_output = self.smpl_model(
                betas=batch['betas'],
                body_pose = torch.zeros((batch_size, 69)).to(self.device),
                global_orient = torch.zeros((batch_size, 3)).to(self.device)
            )
            joints = smpl_output.joints[:, :24]

            if normalise:
                joints = joints / subject_height[:, None, None]

            w = (self.smpl_model.lbs_weights)[None, None].repeat(batch_size, num_frames, 1, 1)

            
            # R, T = sample_cameras(batch_size, num_frames, t)
            R, T = sample_cameras(batch_size, num_frames, self.cfg.DATA)
            R = R.to(self.device)
            T = T.to(self.device)

            # render normal images
            ret = self.renderer(vp, 
                                R, 
                                T, 
                                skinning_weights=w,
                                first_frame_v_cano=batch['first_frame_v_cano'])
            normal_imgs = ret['normals'].permute(0, 1, 4, 2, 3)
            masks = ret['masks'].permute(0, 1, 4, 2, 3)
            skinning_weights_maps = ret['skinning_weights_maps']
            canonical_color_maps = ret['canonical_color_maps']
            
            
            batch['normal_imgs'] = normal_imgs
            batch['R'] = R
            batch['T'] = T
            batch['joints'] = joints
            batch['masks'] = masks
            batch['skinning_weights_maps'] = skinning_weights_maps
            batch['canonical_color_maps'] = canonical_color_maps
            batch['vertex_visibility'] = ret['vertex_visibility']
        return batch 
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='val')
            # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_validation_epoch_end(self):
        pass 
        # names = []
        # vals = []
        # for name, param in self.criterion.named_parameters():
        #     names.append(name.split('.')[-1])
        #     vals.append(torch.exp(-param.data.clone()).item())
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

    #     # Check if skinning weights are enabled in config
    #     if self.cfg.MODEL.SKINNING_WEIGHTS:
    #         # Get the last layer of skinning_head's output_conv2
    #         last_layer = self.model.skinning_head.scratch.output_conv2[-1]
            
    #         if hasattr(last_layer, 'weight') and last_layer.weight.grad is not None:
    #             grad_norm = last_layer.weight.grad.norm().item()
    #             print(f"Skinning head last layer weight gradient norm: {grad_norm}")
                
    #         if hasattr(last_layer, 'bias') and last_layer.bias.grad is not None:
    #             grad_norm = last_layer.bias.grad.norm().item()
    #             print(f"Skinning head last layer bias gradient norm: {grad_norm}")

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             grad_norm = param.grad.norm().item()
    #             if grad_norm > 10:
    #                 print(f"Large gradient in {name}: {grad_norm}")
    
    def test_step(self, batch, batch_idx):
        viridis = plt.colormaps.get_cmap('viridis')

        split = 'test'
        if not self.dev: # more randomness in process_inputs, naive set first batch doesn't work 
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
        mask = batch['masks'].squeeze()
        # global_pose, body_pose = batch['pose'][..., :3], batch['pose'][..., 3:]
        joints = batch['joints']
        normal_maps = batch['normal_imgs']
        w_smpl = batch['skinning_weights_maps']
        vc = batch['canonical_color_maps']


        # ------------------- forward pass -------------------
        preds = self(normal_maps)
        vc_pred, vc_conf = preds['vc'], preds['vc_conf']


        if self.cfg.MODEL.SKINNING_WEIGHTS:
            w_pred, w_conf = preds['w'], preds['w_conf']
            # w_pred = w_smpl + w_pred
        else:
            w_pred = w_smpl
            w_conf = None


        vp_pred, joints_pred = general_lbs(
            vc=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'),
            pose=rearrange(batch['pose'], 'b n c -> (b n) c'),
            lbs_weights=rearrange(w_pred, 'b n h w j -> (b n) (h w) j'),
            J=joints.repeat_interleave(batch['pose'].shape[1], dim=0),
            parents=self.smpl_model.parents 
        )
        vp_pred = rearrange(vp_pred, '(b n) v c -> b n v c', b=B, n=N)
        # joints_pred = rearrange(joints_pred, '(b n) j c -> b n j c', b=B, n=N)


        loss, loss_dict = self.criterion(vp=batch['sampled_posed_points'],
                                         vp_pred=vp_pred,
                                         vc=vc,
                                         vc_pred=vc_pred, 
                                         conf=vc_conf,
                                         mask=mask,
                                         w_pred=w_pred,
                                         w_smpl=w_smpl)

        # Visualise and log
        if self.global_step % self.vis_frequency == 0:
            self.visualiser.visualise(normal_maps=normal_maps.cpu().detach().numpy(),
                                      vp=vp.cpu().detach().numpy(),
                                      vc=vc.cpu().detach().numpy(),
                                      vp_pred=vp_pred.cpu().detach().numpy(),
                                      vc_pred=vc_pred.cpu().detach().numpy(),
                                      conf=vc_conf.cpu().detach().numpy(),
                                      mask=mask.cpu().detach().numpy(),
                                      vertex_visibility=batch['vertex_visibility'].cpu().detach().numpy(),
                                      color=np.argmax(w_pred.cpu().detach().numpy(), axis=-1),
                                      no_annotations=True,
                                      plot_error_heatmap=True)

        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        for k, v in loss_dict.items():
            if k != 'total_loss':
                self.log(f'{split}_{k}', v, on_step=True, on_epoch=False, sync_dist=True, rank_zero_only=True)

        vc_avg_err = torch.linalg.norm(vc - vc_pred, dim=-1) * mask 
        vc_avg_err = vc_avg_err.sum() / mask.sum()
        self.log(f'{split}_vc_avg_dist', vc_avg_err, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)


        # save a bunch of stuff for scenepic visualisation later
        scenepic_dir = os.path.join(self.visualiser.save_dir, 'scenepic')
        if not os.path.exists(scenepic_dir):
            os.makedirs(scenepic_dir)

        
        vp = vp.detach().cpu().numpy()
        vp_pred = vp_pred.detach().cpu().numpy()
        vc_pred = vc_pred.detach().cpu().numpy()
        vc = vc.detach().cpu().numpy()
        color = np.argmax(w_pred.detach().cpu().numpy(), axis=-1)
        normal_maps = normal_maps.detach().cpu().numpy()
        masks = mask.detach().cpu().numpy()
        vertex_visibility = batch['vertex_visibility'].detach().cpu().numpy()
        
        save = False 
        if save:
            np.save(os.path.join(scenepic_dir, f'vp_{self.global_step}.npy'), vp)
            np.save(os.path.join(scenepic_dir, f'vertex_visibility_{self.global_step}.npy'), vertex_visibility)
            np.save(os.path.join(scenepic_dir, f'color_{self.global_step}.npy'), color)
            np.save(os.path.join(scenepic_dir, f'vc_{self.global_step}.npy'), vc)
            np.save(os.path.join(scenepic_dir, f'vc_pred_{self.global_step}.npy'), vc_pred)
            np.save(os.path.join(scenepic_dir, f'vp_pred_{self.global_step}.npy'), vp_pred)
            np.save(os.path.join(scenepic_dir, f'normal_maps_{self.global_step}.npy'), normal_maps)
            np.save(os.path.join(scenepic_dir, f'masks_{self.global_step}.npy'), masks)


        id = 1
        plot_gt = True
        side_by_side = False

        scene = sp.Scene()


        # ----------------------- vc pred -----------------------
        positions = []
        colors = []
        for view in range(4):
            vc_plot = vc_pred[id, view, masks[id, view].astype(np.bool)].reshape(-1, 3)
            vc_plot[..., 0] += 2.0
            positions.append(vc_plot)
            colors.append(color[id, view, masks[id, view].astype(np.bool)].flatten())

        positions = np.concatenate(positions, axis=0)
        colors = np.concatenate(colors, axis=0)
        colors_normalized = colors / colors.max()
        colors_rgb = viridis(colors_normalized)[:, :3] 

        mesh_vc_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vc_pred")
        mesh_vc_pred.add_sphere() 
        mesh_vc_pred.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vc_pred.enable_instancing(positions = positions, colors = colors_rgb) 


        # ----------------------- vc gt -----------------------
        positions = []
        for view in range(4):
            vc_gt = vc[id, view, masks[id, view].astype(np.bool)].reshape(-1, 3)
            vc_gt[..., 0] += 1.2 if side_by_side else 0
            vc_gt[..., 0] += 2.0
            positions.append(vc_gt)
        positions = np.concatenate(positions, axis=0)

        mesh_vc_gt = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vc_gt")
        mesh_vc_gt.add_sphere() 
        mesh_vc_gt.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vc_gt.enable_instancing(positions = positions) 



        # ----------------------- vp pred -----------------------
        positions = []
        colors = []

        for view in range(4):
            vp_plot = vp_pred[id, view, masks[id, view].astype(np.bool).flatten()]
            vp_plot[..., 0] += 0.8 * view - 1.6
            positions.append(vp_plot)
            colors.append(color[id, view, masks[id, view].astype(np.bool)].flatten())


        positions = np.concatenate(positions, axis=0)
        colors = np.concatenate(colors, axis=0)


        colors_normalized = colors / colors.max()
        colors_rgb = viridis(colors_normalized)[:, :3] 

        mesh_vp_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vp_pred")
        mesh_vp_pred.add_sphere() 
        mesh_vp_pred.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vp_pred.enable_instancing(positions = positions, colors = colors_rgb) 


        # ----------------------- vp gt -----------------------
        positions = []
        for view in range(4):
            vp_gt_plot = vp[id, view, vertex_visibility[id, view].astype(np.bool).flatten()]
            vp_gt_plot[..., 0] += 0.8 * view - 1.6
            positions.append(vp_gt_plot)
        positions = np.concatenate(positions, axis=0)

        mesh_vp_gt = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vp_gt")
        mesh_vp_gt.add_sphere() 
        mesh_vp_gt.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vp_gt.enable_instancing(positions = positions) 


        # ----------------------- canvas -----------------------
        golden_ratio = (1 + np.sqrt(5)) / 2
        canvas = scene.create_canvas_3d(width = 1600, height = 1600 / golden_ratio, shading=sp.Shading(bg_color=sp.Colors.White))
        frame = canvas.create_frame()

        frame.add_mesh(mesh_vp_pred)
        frame.add_mesh(mesh_vp_gt)

        frame.add_mesh(mesh_vc_pred)
        frame.add_mesh(mesh_vc_gt)

        path = os.path.join(self.visualiser.save_dir, f'scenepic_vp_{self.global_step}.html')
        scene.save_as_html(path)



        return loss 
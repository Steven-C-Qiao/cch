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
        

    def forward(self, batch, pose=None, joints=None, w_smpl=None, mask=None, R=None, T=None, gt_vc=None):
        return self.model(batch, pose=pose, joints=joints, w_smpl=w_smpl, mask=mask, R=R, T=T, gt_vc=gt_vc)
    
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
        masks = batch['masks'].squeeze()
        # global_pose, body_pose = batch['pose'][..., :3], batch['pose'][..., 3:]
        joints = batch['joints']
        normal_maps = batch['normal_imgs']
        w_smpl = batch['w_pm']
        vc = batch['vc_pm']


        # ------------------- forward pass -------------------
        preds = self(
            normal_maps, 
            pose=batch['pose'], 
            joints=batch['joints'], 
            w_smpl=batch['w_pm'], 
            mask=batch['masks'], 
            R=batch['R'], 
            T=batch['T'],
            gt_vc=vc
        )
        vp_init_pred, vc_pred, w_pred, dvc_pred = preds['vp'], preds['vc'], preds['w'], preds['dvc']
        vc_conf, w_conf, dvc_conf = preds['vc_conf'], preds['w_conf'], None
        vp_cond = preds['vp_cond']
        vp_cond_mask = preds['vp_cond_mask']

        vp_pred, _ = general_lbs(
            vc=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'),
            pose=rearrange(batch['pose'], 'b n c -> (b n) c'),
            lbs_weights=rearrange(w_smpl, 'b n h w j -> (b n) (h w) j'),
            J=joints.repeat_interleave(batch['pose'].shape[1], dim=0),
            parents=self.smpl_model.parents 
        )
        vp_pred = rearrange(vp_pred, '(b n) (h w) c -> b n h w c', b=B, n=N, h=224, w=224)



        loss, loss_dict = self.criterion(
            vp=vp_sampled,
            vp_pred=vp_pred,
            vc=vc,
            vc_pred=vc_pred, 
            conf=vc_conf,
            mask=masks,
            w_pred=w_pred,
            w_smpl=w_smpl,
            dvc_pred=dvc_pred,
            dvc_conf=dvc_conf,
            epoch=self.current_epoch
        )
        self.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)

        self.metrics(vc=rearrange(vc, 'b n h w c -> (b n) (h w) c'), 
                     vc_pred=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'), 
                     vp=rearrange(vp, 'b n v c -> (b n) v c'), 
                     vp_pred=rearrange(vp_pred, 'b n h w c -> (b n) (h w) c'), 
                    #  conf=rearrange(vc_conf, 'b n h w -> (b n) (h w)'), 
                     mask=rearrange(masks, 'b n h w -> (b n) (h w)'), 
                     split=split)

        # Visualise
        if self.global_step % self.vis_frequency == 0:
            self.visualiser.visualise(
                normal_maps=normal_maps.cpu().detach().numpy(),
                vp=vp.cpu().detach().numpy(),
                vc=vc.cpu().detach().numpy(),
                vp_pred=rearrange(vp_pred, 'b n h w c -> b n (h w) c').cpu().detach().numpy(),
                vc_pred=vc_pred.cpu().detach().numpy(),
                # conf=vc_conf.cpu().detach().numpy(),
                mask=masks.cpu().detach().numpy(),
                vertex_visibility=batch['vertex_visibility'].cpu().detach().numpy(),
                color=np.argmax(w_pred.cpu().detach().numpy(), axis=-1),
                dvc=dvc_pred.cpu().detach().numpy(),
                vp_cond=vp_cond.cpu().detach().numpy(),
                vp_cond_mask=masks.cpu().detach().numpy(),
                no_annotations=True,
                plot_error_heatmap=True
            )
        # For app.animate_pm.py 
        # self.save_avatar(vc_pred, vc_conf, w_pred, joints, masks, w_smpl=w_smpl)

        return loss 
    

    def metrics(self, vc=None, vc_pred=None, vp=None, vp_pred=None, conf=None, mask=None, split='train'):
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
        '''
        vp: (B, N, 6890, 3)
        vp_pred: (B, N, H, W, 3)
        mask: (B, N, H, W)
        conf: (B, N, H, W)
        '''
        vp_dist_squared, _ = chamfer_distance(vp_pred, vp, batch_reduction=None, point_reduction=None)
        vpp2vp_dist = torch.sqrt(vp_dist_squared[0])
        masked_vpp2vp_dist = vpp2vp_dist * full_mask
        vpp2vp_dist = masked_vpp2vp_dist.sum() / (full_mask.sum() + 1e-6) * 100.0

        vp2vpp_dist = torch.sqrt(vp_dist_squared[1])
        vp2vpp_dist = vp2vpp_dist.mean() * 100.0


        self.log(f'{split}_vc_pm_dist',          vc_pm_dist,  on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log(f'{split}_vpp2vp_chamfer_dist', vpp2vp_dist, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log(f'{split}_vp2vpp_chamfer_dist', vp2vpp_dist, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)

        return {
            'vc_pm_dist': vc_pm_dist,
            'vpp2vp_chamfer_dist': vpp2vp_dist,
            'vp2vpp_chamfer_dist': vp2vpp_dist,
        }





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

                batch['first_frame_v_cano'] = vc / subject_height[:, None, None] * 1.7 # B, 6890, 3
                batch['v_posed'] = batch['v_posed'] / subject_height[:, None, None, None] * 1.7 # B, N, 6890, 3
                


            smpl_output = self.smpl_model(
                betas=batch['betas'],
                body_pose = torch.zeros((batch_size, 69)).to(self.device),
                global_orient = torch.zeros((batch_size, 3)).to(self.device)
            )
            joints = smpl_output.joints[:, :24]

            # CAPE provided naked shape is bad, 
            # add a correction to the joints by scaling height of the provided naked shape to first_frame_v_cano
            naked_height = (smpl_output.vertices[:, :, 1].max(dim=-1).values - smpl_output.vertices[:, :, 1].min(dim=-1).values)
            joints = joints * (subject_height[:, None, None] / naked_height[:, None, None]) 

            if normalise:
                joints = joints / subject_height[:, None, None] * 1.7

            w = (self.smpl_model.lbs_weights)[None, None].repeat(batch_size, num_frames, 1, 1)

            
            # R, T = sample_cameras(batch_size, num_frames, t)
            R, T = sample_cameras(batch_size, num_frames, self.cfg.DATA)
            R = R.to(self.device)
            T = T.to(self.device)

            # render normal images
            ret = self.renderer(
                vertices=batch['v_posed'], 
                R=R, 
                T=T, 
                skinning_weights=w,
                first_frame_v_cano=batch['first_frame_v_cano']
            )
            normal_imgs = ret['normals'].permute(0, 1, 4, 2, 3)
            masks = ret['masks'].permute(0, 1, 4, 2, 3)
            skinning_weights_maps = ret['skinning_weights_maps']
            canonical_color_maps = ret['canonical_color_maps']
            
            
            batch['normal_imgs'] = normal_imgs
            batch['R'] = R
            batch['T'] = T
            batch['joints'] = joints
            batch['masks'] = masks
            batch['w_pm'] = skinning_weights_maps
            batch['vc_pm'] = canonical_color_maps
            batch['vertex_visibility'] = ret['vertex_visibility']
        return batch 
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='val')
            # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    

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


    def test_step(self, batch, batch_idx):

        import scenepic as sp 
        from matplotlib import pyplot as plt
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
        vp_sampled = batch['v_posed_samples']
        mask = batch['masks'].squeeze()
        # global_pose, body_pose = batch['pose'][..., :3], batch['pose'][..., 3:]
        joints = batch['joints']
        normal_maps = batch['normal_imgs']
        w_smpl = batch['w_pm']
        vc = batch['vc_pm']


        # ------------------- forward pass -------------------
        preds = self(normal_maps, pose=batch['pose'], joints=batch['joints'], w_smpl=batch['w_pm'], mask=batch['masks'], R=batch['R'], T=batch['T'])
        vc_pred, vc_conf = preds['vc'], preds['vc_conf']


        if self.cfg.MODEL.SKINNING_WEIGHTS:
            w_pred, w_conf = preds['w'], preds['w_conf']
        else:
            w_pred, w_conf = w_smpl, None

        if self.cfg.MODEL.POSE_CORRECTIVES:
            dvc_pred, dvc_conf = preds['dvc'], preds['dvc_conf']
            vc_pred = vc_pred + dvc_pred
        else:
            dvc_pred, dvc_conf = None, None


        vp_pred, joints_pred = general_lbs(
            vc=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'),
            pose=rearrange(batch['pose'], 'b n c -> (b n) c'),
            lbs_weights=rearrange(w_pred, 'b n h w j -> (b n) (h w) j'),
            J=joints.repeat_interleave(batch['pose'].shape[1], dim=0),
            parents=self.smpl_model.parents 
        )
        vp_pred = rearrange(vp_pred, '(b n) v c -> b n v c', b=B, n=N)
        # joints_pred = rearrange(joints_pred, '(b n) j c -> b n j c', b=B, n=N)


        loss, loss_dict = self.criterion(vp=vp_sampled,
                                         vp_pred=vp_pred,
                                         vc=vc,
                                         vc_pred=vc_pred, 
                                         conf=vc_conf,
                                         mask=mask,
                                         w_pred=w_pred,
                                         w_smpl=w_smpl,
                                         epoch=self.current_epoch)
        self.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.metrics(vc=rearrange(vc, 'b n h w c -> (b n) (h w) c'), 
                     vc_pred=rearrange(vc_pred, 'b n h w c -> (b n) (h w) c'), 
                     vp=rearrange(vp, 'b n v c -> (b n) v c'), 
                     vp_pred=rearrange(vp_pred, 'b n v c -> (b n) v c'), 
                     conf=rearrange(vc_conf, 'b n h w -> (b n) (h w)'), 
                     mask=rearrange(mask, 'b n h w -> (b n) (h w)'), 
                     split=split)

        # chamfer distance between vc and vc_pred
        vc_first_frame = batch['first_frame_v_cano']

        if vc_conf is not None:
            conf_threshold = 0.08
            conf_mask = (1/vc_conf) < conf_threshold

        full_mask = (mask * conf_mask).bool()
        
        vc_mesh_chamfer, _ = chamfer_distance(rearrange(vc_pred, 'b n h w c -> b (n h w) c'), 
                                           vc_first_frame, 
                                           batch_reduction=None, point_reduction=None)

        vc_mesh_chamfer = vc_mesh_chamfer[0] * rearrange(full_mask, 'b n h w -> b (n h w)')
        vc_mesh_chamfer = vc_mesh_chamfer.sum() / (full_mask.sum() + 1e-6)

        self.log(f'{split}_vc_mesh_chamfer', vc_mesh_chamfer, sync_dist=True, rank_zero_only=True)


        save_scenepic = True
        if save_scenepic:
            save_scenepic = False
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
    

    def save_avatar(self, vc_pred, vc_conf, w_pred, joints, masks, w_smpl):
        save_avatar = True
        if save_avatar:
            import pickle
            with open('tinkering/avatar.pkl', 'wb') as f:
                pickle.dump({
                    # 'normal_maps': normal_maps,
                    # 'vertex_visibility': vertex_visibility,
                    # 'color': color,
                    'masks': masks.cpu().detach().numpy(),
                    'vc_pred': vc_pred.cpu().detach().numpy(),
                    'vc_conf': vc_conf.cpu().detach().numpy(),
                    'w_pred': w_pred.cpu().detach().numpy(),
                    'w_smpl': w_smpl.cpu().detach().numpy(),
                    'joints': joints.cpu().detach().numpy()
                }, f)
            print(vc_pred.shape, vc_conf.shape, w_pred.shape, joints.shape)

        import ipdb; ipdb.set_trace()
        print('')


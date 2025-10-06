
import torch
import numpy as np

import torch.optim as optim
import pytorch_lightning as pl

from einops import rearrange

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from smplx.lbs import batch_rodrigues


from core.configs import paths 

from core.models.smpl import SMPL
from hmr.hmr_baseline import HMR 

from hmr.hmr_loss import HMRLoss

from core.utils.renderer import SurfaceNormalRenderer 
from core.utils.sample_utils import sample_cameras
# from core.utils.visualiser import Visualiser

from pytorch3d.loss import chamfer_distance

class HMRTrainer(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 dev=False, 
                 vis_save_dir=None):
        
        super().__init__()
        self.dev = dev
        self.cfg = cfg
        self.normalise = cfg.DATA.NORMALISE
        self.vis_frequency = cfg.VISUALISE_FREQUENCY if not dev else 100

        self.vis_save_dir = vis_save_dir
 
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

        self.model = HMR(
            smpl_model=self.smpl_model
        )

        self.criterion = HMRLoss()
        # self.visualiser = Visualiser(save_dir=vis_save_dir)

        self.save_hyperparameters(ignore=['smpl_model'])

        self.pve = []
        self.pve_t = []
        self.chamfer_loss = []
        self.chamfer_loss_t = []

        self.test_step_count = 0
        

    def forward(self, batch):
        return self.model(batch)
    
    def on_train_epoch_start(self):
        pass
        # self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train', plot=True):
        process_func = self.process_inputs if self.cfg.DATA.TYPE == 'smpl' else self.process_cape_inputs
        if not self.dev: # more randomness in process_inputs, naive set first batch doesn't work 
            batch = process_func(batch, batch_idx, normalise=self.normalise)
        if self.global_step == 0:
            if self.dev:
                batch = process_func(batch, batch_idx, normalise=self.normalise)
            self.first_batch = batch
            # self.visualiser.visualise_input_normal_imgs(batch['normal_imgs'])
        if self.dev:    
            batch = self.first_batch

        B, N = batch['pose'].shape[:2]
        vp = batch['v_posed']
        vc = batch['v_cano']
        mask = batch['masks'].squeeze()
        # global_pose, body_pose = batch['pose'][..., :3], batch['pose'][..., 3:]
        normal_maps = batch['normal_imgs']


        # ------------------- forward pass -------------------
        preds = self(normal_maps)
        pred_pose, pred_glob_orient = preds['pred_rotmat'][:, 1:], preds['pred_rotmat'][:, [0]]

        # for k, v in preds.items():
        #     print(k, v.shape)

            # pred_shape torch.Size([2, 10])
            # pred_rotmat torch.Size([8, 24, 3, 3])
            # pred_cam torch.Size([8, 3])

        smpl_output = self.smpl_model(
            betas=preds['pred_shape'].repeat_interleave(N, dim=0),
            body_pose=pred_pose,
            global_orient=pred_glob_orient,
            pose2rot=False
        )
        vp_pred = smpl_output.vertices
        vp_pred = rearrange(vp_pred, '(b n) v c -> b n v c', b=B, n=N)

        t_smpl_pred_output = self.smpl_model(
            betas=preds['pred_shape'],
            body_pose=torch.zeros(B, 69).to(self.device),
            global_orient=torch.zeros(B, 3).to(self.device),
            pose2rot=True
        )
        vc_pred = t_smpl_pred_output.vertices[:, None].repeat(1, N, 1, 1)


        gt_pose_rotmats = batch_rodrigues(batch['pose'].reshape(-1, 3))
        gt_pose_rotmats = gt_pose_rotmats.view(-1, 24, 3, 3)[:, 1:]


        loss, loss_dict = self.criterion(vp=batch['v_posed'].view(-1, 6890, 3),
                                         vc=batch['v_cano'].view(-1, 6890, 3),
                                         vp_pred=vp_pred.view(-1, 6890, 3),
                                         vc_pred=vc_pred.view(-1, 6890, 3),
                                         pose_pred=pred_pose,
                                         pose_gt=gt_pose_rotmats)
        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        for k, v in loss_dict.items():
            if k != 'total_loss':
                self.log(f'{split}_{k}', v, on_step=True, on_epoch=False, sync_dist=True)

        metrics = self.metrics(vp_pred.view(-1, 6890, 3), 
                                vc_pred.view(-1, 6890, 3), 
                                vp.view(-1, 6890, 3), 
                                vc.view(-1, 6890, 3))
        self.pve.append(metrics['pve'].item())
        self.pve_t.append(metrics['pve_t'].item())
        self.chamfer_loss.append(metrics['chamfer_loss'].item())
        self.chamfer_loss_t.append(metrics['chamfer_loss_t'].item())

        if plot and self.global_step % 1000 == 0 and self.global_rank == 0:
            self.visualise(vp_pred=vp_pred.cpu().detach().numpy(), 
                            vc_pred=vc_pred.cpu().detach().numpy(), 
                            vp=vp.cpu().detach().numpy(), 
                            vc=vc.cpu().detach().numpy())
        # self.test_step_count += 1
        return loss 



    def process_inputs(self, batch, batch_idx, normalise=False):
        """
        Batch of size B contains N frames of the same person. For each frame, sample a random camera pose and render.
        """
        with torch.no_grad():
            B, N = batch['pose'].shape[:2]

            betas = torch.randn(B, 10).to(self.device)
            smpl_output = self.smpl_model(betas=betas.repeat_interleave(N, dim=0), 
                                        body_pose=batch['pose'].view(-1, 69), 
                                        global_orient=torch.zeros(B * N, 3).to(self.device))
            
            
            batch['v_posed'] = rearrange(smpl_output.vertices, '(b n) v c -> b n v c', b=B, n=N)
            batch['joints'] = rearrange(smpl_output.joints[:, :24], '(b n) j c -> b n j c', b=B, n=N)

            t_smpl_output = self.smpl_model(betas=betas, 
                                        body_pose=torch.zeros(B, 69).to(self.device), 
                                        global_orient=torch.zeros(B, 3).to(self.device))
            batch['v_cano'] = t_smpl_output.vertices
            batch['joints_cano'] = t_smpl_output.joints[:, :24]
            vp = batch['v_posed'] # B, N, 6890, 3

            batch_size, num_frames = vp.shape[:2]


            w = (self.smpl_model.lbs_weights)[None, None].repeat(batch_size, num_frames, 1, 1)

            
            R, T = sample_cameras(batch_size, num_frames, self.cfg.DATA)
            R = R.to(self.device)
            T = T.to(self.device)

            # render normal images
            ret = self.renderer(vp, 
                                R, 
                                T, 
                                skinning_weights=w,
                                first_frame_v_cano=batch['v_cano'])
            normal_imgs = ret['normals'].permute(0, 1, 4, 2, 3)
            masks = ret['masks'].permute(0, 1, 4, 2, 3)
            
            # canonical_color_maps = torch.tensor(ret['canonical_color_maps'], 
            #                                     dtype=torch.float32).to(self.device)
            
            
            batch['normal_imgs'] = normal_imgs
            batch['R'] = R
            batch['T'] = T
            batch['masks'] = masks
            batch['skinning_weights_maps'] = ret['skinning_weights_maps']
            batch['canonical_color_maps'] = ret['canonical_color_maps']
            batch['vertex_visibility'] = ret['vertex_visibility']
        return batch 
    

    def process_cape_inputs(self, batch, batch_idx, normalise=False):
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

            R, T = sample_cameras(batch_size, num_frames, self.cfg.DATA)
            R = R.to(self.device)
            T = T.to(self.device)

            # render normal images
            ret = self.renderer(vp, 
                                R, 
                                T)
            normal_imgs = ret['normals'].permute(0, 1, 4, 2, 3)
            masks = ret['masks'].permute(0, 1, 4, 2, 3)
            
            
            batch['normal_imgs'] = normal_imgs
            batch['R'] = R
            batch['T'] = T
            batch['masks'] = masks
        return batch 
    

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='val')
            # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx, split='test', plot=False)
            # self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
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
        
        
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)


    def visualise(self, vp_pred, vc_pred, vp, vc, no_annotations=False):
        s = 0.04
        alpha = 0.5
        B, N = vp_pred.shape[:2]

        B = min(B, 2)
        N = min(N, 4)

        fig = plt.figure(figsize=(N*4, B*4))
        for b in range(B):
            for n in range(N):
                vp_pred_b_n = vp_pred[b, n]
                vc_pred_b_n = vc_pred[b, n]
                vp_b_n = vp[b, n]
                vc_b_n = vc[b, n]

                # Create 3D scatter plot of predicted vertices
                ax = fig.add_subplot(B, N, b*N + n + 1, projection='3d')
                ax.scatter(vp_pred_b_n[:,0], vp_pred_b_n[:,1], vp_pred_b_n[:,2], s=s, alpha=alpha, color='red', label='Pred')
                ax.scatter(vp_b_n[:,0], vp_b_n[:,1], vp_b_n[:,2], s=s, alpha=alpha, color='blue', label='GT')


                ax.set_box_aspect([1,1,1])
                
                max_range = np.array([
                    vp_pred_b_n[:,0].max() - vp_pred_b_n[:,0].min(),
                    vp_pred_b_n[:,1].max() - vp_pred_b_n[:,1].min(),
                    vp_pred_b_n[:,2].max() - vp_pred_b_n[:,2].min()
                ]).max() / 2.0
                mid_x = (vp_pred_b_n[:,0].max() + vp_pred_b_n[:,0].min()) * 0.5
                mid_y = (vp_pred_b_n[:,1].max() + vp_pred_b_n[:,1].min()) * 0.5
                mid_z = (vp_pred_b_n[:,2].max() + vp_pred_b_n[:,2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

                ax.view_init(elev=10, azim=20, vertical_axis='y')

                if not no_annotations:
                    ax.set_title(f'Vp (batch {b}, view {n})')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y') 
                    ax.set_zlabel('Z')

    
        if no_annotations:
            for ax in fig.axes:
                ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('none')
                ax.yaxis.pane.set_edgecolor('none') 
                ax.zaxis.pane.set_edgecolor('none')
                ax.xaxis.line.set_color('none')
                ax.yaxis.line.set_color('none')
                ax.zaxis.line.set_color('none')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])


        plt.tight_layout(pad=0.01)
        plt.savefig(f'{self.vis_save_dir}/{self.global_step}_vp.png', dpi=300)
        plt.close()

        fig = plt.figure(figsize=(3*4, B*4))
        for b in range(B):
            vc_b = vc[b, 0]
            vc_pred_b = vc_pred[b, 0]

            ax = fig.add_subplot(B, 3, b*3 + 1, projection='3d')
            ax.scatter(vc_b[:,0], vc_b[:,1], vc_b[:,2], s=s, alpha=alpha, color='blue', label='GT')
            ax.scatter(vc_pred_b[:,0], vc_pred_b[:,1], vc_pred_b[:,2], s=s, alpha=alpha, color='red', label='Pred')
            ax.legend()

            ax = fig.add_subplot(B, 3, b*3 + 2, projection='3d')
            ax.scatter(vc_b[:,0], vc_b[:,1], vc_b[:,2], s=s, alpha=alpha, color='blue')

            ax = fig.add_subplot(B, 3, b*3 + 3, projection='3d')
            ax.scatter(vc_pred_b[:,0], vc_pred_b[:,1], vc_pred_b[:,2], s=s, alpha=alpha, color='red')


        max_range = np.array([
            vc_b[:,0].max() - vc_b[:,0].min(),
            vc_b[:,1].max() - vc_b[:,1].min(),
            vc_b[:,2].max() - vc_b[:,2].min()
        ]).max() / 2.0
        mid_x = (vc_b[:,0].max() + vc_b[:,0].min()) * 0.5
        mid_y = (vc_b[:,1].max() + vc_b[:,1].min()) * 0.5
        mid_z = (vc_b[:,2].max() + vc_b[:,2].min()) * 0.5

        for ax in fig.axes:
            ax.set_box_aspect([1,1,1])
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.view_init(elev=10, azim=20, vertical_axis='y')

            if not no_annotations:
                ax.set_title(f'Vc (batch {b})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y') 
                ax.set_zlabel('Z')

        
    
        if no_annotations:
            for ax in fig.axes:
                ax.grid(False)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('none')
                ax.yaxis.pane.set_edgecolor('none') 
                ax.zaxis.pane.set_edgecolor('none')
                ax.xaxis.line.set_color('none')
                ax.yaxis.line.set_color('none')
                ax.zaxis.line.set_color('none')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

        
        plt.tight_layout(pad=0.01)
        plt.savefig(f'{self.vis_save_dir}/{self.global_step}_vc.png', dpi=300)
        plt.close()



    def metrics(self, vp_pred, vc_pred, vp, vc):
        B, N = vp_pred.shape[:2]

        # PVE
        pve = torch.norm(vp_pred - vp, dim=-1)
        pve = pve.mean()
        self.log('pve', pve, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # PVE-T
        pve_t = torch.norm(vc_pred - vc, dim=-1)
        pve_t = pve_t.mean()
        self.log('pve_t', pve_t, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # chamfer 
        chamfer_loss, _ = chamfer_distance(vp_pred, vp,
                                           batch_reduction=None, point_reduction=None)
        chamfer_loss_v_pred_to_v = chamfer_loss[0]
        chamfer_loss_v_to_v_pred = chamfer_loss[1]

        chamfer_loss_v_pred_to_v = chamfer_loss_v_pred_to_v.mean()
        chamfer_loss_v_to_v_pred = chamfer_loss_v_to_v_pred.mean()

        self.log('chamfer_loss_v_pred_to_v', chamfer_loss_v_pred_to_v, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('chamfer_loss_v_to_v_pred', chamfer_loss_v_to_v_pred, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        chamfer_loss = chamfer_loss_v_pred_to_v + chamfer_loss_v_to_v_pred
        self.log('chamfer_loss', chamfer_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # chamfer_t
        chamfer_loss_t, _ = chamfer_distance(vc_pred, vc)
        self.log('chamfer_loss_t', chamfer_loss_t, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        # import ipdb; ipdb.set_trace()

        return {
            'pve': pve,
            'pve_t': pve_t,
            'chamfer_loss': chamfer_loss,
            'chamfer_loss_t': chamfer_loss_t
        }


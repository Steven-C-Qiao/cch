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
from core.utils.feature_renderer import FeatureRenderer

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
 
        # self.feature_renderer = FeatureRenderer(image_size=(224, 224))
        self.feature_renderer = FeatureRenderer(image_size=(1280, 940))

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
        vc_init_pred, vc_pred,  vc_init_pred_conf = preds['vc_init_pred'], preds['vc_pred'], preds['vc_conf_init_pred']
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


    @torch.no_grad()
    def process_inputs(self, batch, batch_idx, normalise=False):
        B, N = batch['imgs'].shape[:2]

        # get joint positions
        smpl_T_output = self.smpl_model(
            betas=batch['betas'].view(-1, 10),
            body_pose = torch.zeros((B*N, 69)).to(self.device),
            global_orient = torch.zeros((B*N, 3)).to(self.device),
            transl = torch.zeros((B*N, 3)).to(self.device)
        )
        smpl_T_joints = smpl_T_output.joints[:, :24].view(B, N, 24, 3)
        smpl_T_vertices = smpl_T_output.vertices.view(B, N, 6890, 3)

        smpl_output = self.smpl_model(
            betas=batch['betas'].view(-1, 10),
            body_pose = batch['body_pose'].view(-1, 69),
            global_orient = batch['global_orient'].view(-1, 3),
            transl = batch['transl'].view(-1, 3)
        )
        smpl_joints = smpl_output.joints[:, :24].view(B, N, 24, 3)
        smpl_vertices = smpl_output.vertices.view(B, N, 6890, 3)
        smpl_skinning_weights = (self.smpl_model.lbs_weights)[None, None].repeat(B, N, 1, 1)

        scan_meshes = batch['scan_mesh']
        scan_skinning_weights = []
        
        # Assign SMPL skinning weights to scan meshes
        for i in range(B):
            batch_scan_weights = []
            for j in range(N):
                scan_mesh = scan_meshes[i][j]
                smpl_verts = smpl_vertices[i, j]  # (6890, 3)
                scan_verts = torch.tensor(scan_mesh['vertices'], dtype=torch.float32, device=self.device)  # (n_scan_verts, 3)
                scan_faces = torch.tensor(scan_mesh['faces'], dtype=torch.int32, device=self.device)
                
                # Compute distances between scan vertices and SMPL vertices
                # Use broadcasting to compute all pairwise distances efficiently
                # scan_verts: (n_scan_verts, 1, 3), smpl_verts: (1, 6890, 3)
                distances = torch.norm(scan_verts.unsqueeze(1) - smpl_verts.unsqueeze(0), dim=2)  # (n_scan_verts, 6890)
                
                # Find nearest SMPL vertex for each scan vertex
                nearest_indices = torch.argmin(distances, dim=1)  # (n_scan_verts,)
                
                # Get skinning weights for the nearest SMPL vertices
                nearest_weights = smpl_skinning_weights[i, j][nearest_indices]  # (n_scan_verts, 24)
                
                # Store the skinning weights for this scan mesh
                # batch_scan_weights.append(nearest_weights)
                # print(nearest_weights.shape)
                scan_skinning_weights.append(nearest_weights)
            
            # scan_skinning_weights.append(batch_scan_weights)
        
        # Add scan skinning weights to batch
        batch['scan_skinning_weights'] = scan_skinning_weights

        # build pytorch3d cameras
        R, T, K = batch['R'], batch['T'], batch['K']
        from pytorch3d.renderer import PerspectiveCameras
        # print(T.shape)
        # print(R.shape)
        # print(K.shape)
        R = R.view(-1, 3, 3).float()
        T = T.view(-1, 3).float()
        K = K.view(-1, 4, 4).float()
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            K=K,
            image_size=[(1280, 940)],
            device=self.device,
            in_ndc=False
        )
        self.feature_renderer._set_cameras(cameras)
        
        

        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex

        verts = [v for sublist in batch['scan_mesh_verts'] for v in sublist]
        faces = [f for sublist in batch['scan_mesh_faces'] for f in sublist]

        pytorch3d_mesh = Meshes(
            verts=verts,
            faces=faces,
            textures=TexturesVertex(verts_features=scan_skinning_weights)
        ).to(self.device)


        renderer_ret = self.feature_renderer(pytorch3d_mesh)
        batch['smpl_w_maps'] = renderer_ret['maps']






        # pytorch3d_mesh = Meshes(
        #     verts=batch['template_mesh_verts'],
        #     faces=batch['template_mesh_faces'],
        # ).to(self.device)
        # verts_padded = pytorch3d_mesh.verts_padded()
        # faces_padded = pytorch3d_mesh.faces_padded()
        # verts_padded_interleaved = verts_padded.repeat_interleave(N, dim=0)
        # faces_padded_interleaved = faces_padded.repeat_interleave(N, dim=0)
        # interleaved_pytorch3d_mesh = Meshes(
        #     verts=verts_padded_interleaved,
        #     faces=faces_padded_interleaved,
        #     textures=TexturesVertex(verts_features=verts_padded_interleaved)
        # ).to(self.device)

        # renderer_ret = self.feature_renderer(interleaved_pytorch3d_mesh)
        # print(renderer_ret['vc_maps'].shape)

        # Visualize rendered maps in a 2x4 grid
        to_vis = torch.argmax(renderer_ret['maps'], dim=-1).cpu().detach().numpy()
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(2):
            for j in range(4):
                idx = i * 4 + j
                axes[i,j].imshow(to_vis[idx])
                axes[i,j].axis('off')
        
        plt.tight_layout()
        plt.savefig('rendered_maps.png')
        plt.close()

        import ipdb; ipdb.set_trace()

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



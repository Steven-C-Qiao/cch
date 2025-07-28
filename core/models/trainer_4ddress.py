import torch
import torch.optim as optim
import pytorch_lightning as pl
from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.knn import knn_points
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from core.configs import paths 
from core.models.smpl import SMPL
from core.models.cch import CCH 
from core.losses.cch_loss import CCHLoss
from core.losses.cch_metrics import CCHMetrics
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
        self.feature_renderer = FeatureRenderer(image_size=(256, 188))

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
        self.metrics = CCHMetrics(cfg)
        self.visualiser = Visualiser(save_dir=vis_save_dir)

        self.save_hyperparameters(ignore=['smpl_model'])

        self.first_batch = None
        

    def forward(self, images, pose=None, joints=None, w_smpl=None, mask=None):
        return self.model(images, pose=pose, joints=joints, w_smpl=w_smpl, mask=mask)
    
    def on_train_epoch_start(self):
        self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train'):
        if self.first_batch is None:
            self.first_batch = batch
        if self.dev:
            batch = self.first_batch

        batch = self._process_inputs(batch, batch_idx, normalise=self.normalise)
    
        preds = self(
            images=batch['imgs'],
            pose=batch['pose'], 
            joints=batch['smpl_T_joints'], 
            w_smpl=batch['smpl_w_maps'], 
            mask=batch['masks']
        )

        loss, loss_dict = self.criterion(preds, batch)

        metrics = self.metrics(preds, batch)

        self._log_metrics_and_visualise(loss, loss_dict, metrics, split, preds, batch, self.global_step)
        
        return loss 
    

    def _log_metrics_and_visualise(self, loss, loss_dict, metrics, split, preds, batch, global_step):
        self.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

        # if (global_step % self.vis_frequency == 0 and global_step > 0) or (global_step == 1):
        #     self.visualiser.visualise(preds, batch) 

        if self.dev:
            self.visualiser.visualise(preds, batch)
            import ipdb; ipdb.set_trace()
    


    @torch.no_grad()
    def _process_inputs(self, batch, batch_idx, normalise=False):

        B, N = batch['imgs'].shape[:2]

        # ----------------------- get T joints -----------------------
        smpl_T_output = self.smpl_model(
            betas=batch['betas'].view(-1, 10),
            body_pose = torch.zeros((B*N, 69)).to(self.device),
            global_orient = torch.zeros((B*N, 3)).to(self.device),
            transl = torch.zeros((B*N, 3)).to(self.device)
        )
        smpl_faces = torch.tensor(self.smpl_model.faces, dtype=torch.int32)
        smpl_T_joints = smpl_T_output.joints[:, :24].view(B, N, 24, 3)
        smpl_T_vertices = smpl_T_output.vertices.view(B, N, 6890, 3)
        batch['smpl_T_joints'] = smpl_T_joints

        smpl_output = self.smpl_model(
            betas=batch['betas'].view(-1, 10),
            body_pose = batch['body_pose'].view(-1, 69),
            global_orient = batch['global_orient'].view(-1, 3),
            transl = batch['transl'].view(-1, 3)
        )
        smpl_vertices = smpl_output.vertices.view(B, N, 6890, 3)
        smpl_skinning_weights = (self.smpl_model.lbs_weights)[None, None].repeat(B, N, 1, 1)


        scan_meshes = batch['scan_mesh']
        scan_mesh_verts = [v for sublist in batch['scan_mesh_verts'] for v in sublist]
        scan_mesh_faces = [f for sublist in batch['scan_mesh_faces'] for f in sublist]


        dists, idx = self.knn_ptcld(
            Pointclouds(points=scan_mesh_verts), 
            smpl_vertices.view(-1, 6890, 3), 
            K=1
        )
        smpl_weights_flat = smpl_skinning_weights.view(-1, 6890, 24)
        idx_expanded = idx.repeat(1, 1, 24)
        scan_w_tensor = torch.gather(smpl_weights_flat, dim=1, index=idx_expanded)
        scan_w = [scan_w_tensor[i, :len(verts), :] for i, verts in enumerate(scan_mesh_verts)]
        batch['scan_skinning_weights'] = scan_w


        # build pytorch3d cameras
        R, T, K = batch['R'], batch['T'], batch['K']
        R = R.view(-1, 3, 3).float()
        T = T.view(-1, 3).float()
        K = K.view(-1, 4, 4).float()
        cameras = PerspectiveCameras(
            R=R, T=T, K=K,
            image_size=[(1280, 940)],
            device=self.device,
            in_ndc=False
        )
        self.feature_renderer._set_cameras(cameras)



        # temp_mask = batch['masks']
        # B, N, H, W, _ = temp_mask.shape
        # temp_mask = rearrange(temp_mask, 'b n h w c -> (b n) h w c', b=B, n=N)
        # target_size = W 
        # crop_amount = (H - target_size) // 2  
        # temp_mask = temp_mask[:, crop_amount:H-crop_amount, :, :]
        # temp_mask = torch.nn.functional.interpolate(temp_mask.permute(0,3,1,2), size=(224,224), mode='bilinear', align_corners=False)
        # temp_mask = rearrange(temp_mask, '(b n) j h w -> b n h w j', b=B, n=N)
        


        pytorch3d_mesh = Meshes(
            verts=scan_mesh_verts,
            faces=scan_mesh_faces,
            textures=TexturesVertex(verts_features=scan_w)
        ).to(self.device)

        w_maps = self.feature_renderer(pytorch3d_mesh)['maps']
        _, H, W, _ = w_maps.shape
        target_size = W 
        crop_amount = (H - target_size) // 2  
        w_maps = w_maps[:, crop_amount:H-crop_amount, :, :]
        w_maps = torch.nn.functional.interpolate(w_maps.permute(0,3,1,2), size=(224,224), mode='bilinear', align_corners=False)
        w_maps = rearrange(w_maps, '(b n) j h w -> b n h w j', b=B, n=N)
        
        batch['smpl_w_maps'] = w_maps


        smpl_pytorch3d_mesh = Meshes(
            verts=smpl_vertices.view(-1, 6890, 3),
            faces=smpl_faces[None].repeat(B*N, 1, 1).to(self.device),
            textures=TexturesVertex(verts_features=smpl_T_vertices.view(-1, 6890, 3))
        ).to(self.device)
        ret = self.feature_renderer(smpl_pytorch3d_mesh)
        vc_maps = ret['maps']
        mask = ret['mask'].unsqueeze(-1)
        _, H, W, _ = vc_maps.shape
        target_size = W 
        crop_amount = (H - target_size) // 2  
        vc_maps = vc_maps[:, crop_amount:H-crop_amount, :, :]
        vc_maps = torch.nn.functional.interpolate(vc_maps.permute(0,3,1,2), size=(224,224), mode='bilinear', align_corners=False)
        vc_maps = rearrange(vc_maps, '(b n) c h w -> b n h w c', b=B, n=N)
        batch['vc_maps'] = vc_maps

        mask = mask[:, crop_amount:H-crop_amount, :, :]
        mask = torch.nn.functional.interpolate(mask.permute(0,3,1,2), size=(224,224), mode='bilinear', align_corners=False)
        mask = rearrange(mask, '(b n) c h w -> b n h w c', b=B, n=N)
        batch['smpl_mask'] = mask.squeeze(-1)



        vp = [v for sublist in batch['scan_mesh_verts_centered'] for v in sublist]
        vp_ptcld = Pointclouds(points=vp)
        batch['vp_ptcld'] = vp_ptcld


        # self._test_smpl_scan_alignment(smpl_vertices, scan_mesh_verts)
        # self._test_render(vc_maps, masks=mask, name='vc_maps')
        # self._test_render(w_maps, name='w_maps')

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


    def knn_ptcld(self, x, y, K=1):
        from pytorch3d.loss.chamfer import _handle_pointcloud_input
        from pytorch3d.ops.knn import knn_points

        x_lengths, x_normals = None, None
        y_lengths, y_normals = None, None

        x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
        y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=2, K=K)
        dist, idx, _ = x_nn
        return dist, idx
   

    def _test_smpl_scan_alignment(self, smpl_vertices, scan_mesh_verts):
        # Create scatter plot of SMPL and scan vertices
        import matplotlib.pyplot as plt



        # Create 3D figure
        fig = plt.figure(figsize=(12, 6))
        
        # SMPL vertices plot
        ax1 = fig.add_subplot(121, projection='3d')
        smpl_verts = smpl_vertices[0,1].detach().cpu().numpy()
        ax1.scatter(smpl_verts[:,0], smpl_verts[:,1], smpl_verts[:,2], s=0.1, c='blue', marker='.', alpha=0.6)
        ax1.set_title('SMPL Vertices')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y') 
        ax1.set_zlabel('Z')

        # Scan vertices plot
        ax2 = fig.add_subplot(122, projection='3d')
        scan_verts = scan_mesh_verts[1].detach().cpu().numpy()
        ax2.scatter(smpl_verts[:,0], smpl_verts[:,1], smpl_verts[:,2], s=0.1, c='blue', marker='.', alpha=0.6)
        ax2.scatter(scan_verts[:,0], scan_verts[:,1], scan_verts[:,2], s=0.1, c='red', marker='.', alpha=0.6)
        ax2.set_title('Scan Vertices')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.tight_layout()
        plt.show()
        plt.close()

    def _test_render(self, maps, masks=None, name=None):
        if maps.shape[-1] >= 4:
            # Visualize rendered maps in a 2x4 grid
            to_vis = torch.argmax(maps, dim=-1).cpu().detach().numpy()
        else:
            assert maps.shape[-1] == 3
            to_vis = maps.cpu().detach().numpy()

        if masks is not None:
            mask = masks.cpu().detach().numpy().squeeze()
            to_vis[~mask.astype(bool)] = 0
            norm_min, norm_max = to_vis.min(), to_vis.max()
            to_vis = (to_vis - norm_min) / (norm_max - norm_min) 
            to_vis[~mask.astype(bool)] = 1

        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(2):
            for j in range(4):
                idx = i * 4 + j
                axes[i,j].imshow(to_vis[i, j])
                axes[i,j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{name}_rendered_maps.png')
        plt.close()

        # import ipdb; ipdb.set_trace()

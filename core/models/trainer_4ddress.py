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

        batch = self.process_inputs(batch, batch_idx, normalise=self.normalise)


        preds = self(
            images=batch['imgs'],
            pose=batch['pose'], 
            joints=batch['smpl_T_joints'], 
            w_smpl=batch['smpl_w_maps'], 
            mask=batch['masks']
        )
        

        loss, loss_dict = self.criterion(preds, batch)

        metrics = self.metrics(preds, batch)


        self.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

        # if (self.global_step % self.vis_frequency == 0 and self.global_step > 0) or (self.global_step == 1):
        self.visualiser.visualise(preds, batch)
        import ipdb; ipdb.set_trace()


        # self.metrics(
        #     # vc=rearrange(vc, 'b n h w c -> (b n) (h w) c'), 
        #     # vc_pred=rearrange(vc_init_pred, 'b n h w c -> (b n) (h w) c'), 
        #     vp=batch['vp_ptcld'], 
        #     vp_pred=rearrange(preds['vp_pred'], 'b k n h w c -> (b k) (n h w) c'), 
        #     conf=rearrange(preds['vc_conf_init_pred'], 'b n h w -> (b n) (h w)'), 
        #     mask=rearrange(batch['masks'], 'b n h w -> (b n) (h w)'), 
        #     split=split, 
        #     B=B,
        #     N=N
        # )


        # Convert predictions to numpy if tensor
        # preds_np = {}
        # for k, v in preds.items():
        #     if isinstance(v, torch.Tensor):
        #         preds_np[k] = v.cpu().detach().numpy()
        #     else:
        #         preds_np[k] = v

        # # Convert batch tensors to numpy
        # batch_np = {}
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         batch_np[k] = v.cpu().detach().numpy()
        #     else:
        #         batch_np[k] = v
        

        # Visualise
        # if (self.global_step % self.vis_frequency == 0 and self.global_step > 0) or (self.global_step == 1):
        # self.visualiser.visualise(
        #         # normal_maps=normal_maps.cpu().detach().numpy(),
        #         vp=batch['vp_ptcld'].points_padded().view(B, N, -1, 3).cpu().detach().numpy(),
        #         # vc=vc.cpu().detach().numpy(),
        #         vp_pred=preds['vp_pred'].cpu().detach().numpy(),
        #         vp_init_pred=preds['vp_init_pred'].cpu().detach().numpy(),
        #         vc_pred=preds['vc_pred'].cpu().detach().numpy(),
        #         vc_init_pred=preds['vc_init_pred'].cpu().detach().numpy(),
        #         conf=preds['vc_conf_init_pred'].cpu().detach().numpy(),
        #         mask=batch['masks'].cpu().detach().numpy(),
        #         # vertex_visibility=preds['vertex_visibility'].cpu().detach().numpy(),
        #         # color=np.argmax(preds['w_pred'].cpu().detach().numpy(), axis=-1),
        #         # dvc=dvc_pm_target.cpu().detach().numpy(),
        #         dvc_pred=preds['dvc_pred'].cpu().detach().numpy(),
        #         no_annotations=True,
        #         plot_error_heatmap=True
        #     )


        # For app.animate_pm.py 
        # self.save_avatar(vc_pred, vc_conf, w_pred, joints, masks, w_smpl=w_smpl)

        return loss 
    


    @torch.no_grad()
    def process_inputs(self, batch, batch_idx, normalise=False):
        B, N = batch['imgs'].shape[:2]

        # ----------------------- get T joints -----------------------
        smpl_T_output = self.smpl_model(
            betas=batch['betas'].view(-1, 10),
            body_pose = torch.zeros((B*N, 69)).to(self.device),
            global_orient = torch.zeros((B*N, 3)).to(self.device),
            transl = torch.zeros((B*N, 3)).to(self.device)
        )
        smpl_T_joints = smpl_T_output.joints[:, :24].view(B, N, 24, 3)
        smpl_T_vertices = smpl_T_output.vertices.view(B, N, 6890, 3)
        batch['smpl_T_joints'] = smpl_T_joints

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
        scan_mesh_verts = [v for sublist in batch['scan_mesh_verts'] for v in sublist]
        scan_mesh_faces = [f for sublist in batch['scan_mesh_faces'] for f in sublist]

        scan_mesh_verts_ptcld = Pointclouds(points=scan_mesh_verts)


        dists, idx = self.knn_ptcld(scan_mesh_verts_ptcld, smpl_output.vertices, K=1)
        
        # Get skinning weights for each scan vertex based on nearest SMPL vertex
        # smpl_output.vertices has shape (B*N, 6890, 3)
        # smpl_skinning_weights has shape (B, N, 6890, 24)
        # idx has shape (num_scan_clouds, num_scan_vertices, 1)
        
        # Reshape smpl_skinning_weights to (B*N, 6890, 24) to match smpl_output.vertices
        smpl_weights_flat = smpl_skinning_weights.view(-1, 6890, 24)
        
        # For each scan vertex, get the skinning weights of its nearest SMPL vertex
        # idx has shape (num_scan_clouds, num_scan_vertices, 1)
        # We need to expand idx to (num_scan_clouds, num_scan_vertices, 24) for gathering
        idx_expanded = idx.repeat(1, 1, 24)
        
        # Gather the skinning weights using the nearest neighbor indices
        scan_w_tensor = torch.gather(smpl_weights_flat, dim=1, index=idx_expanded)

        scan_w = [scan_w_tensor[i, :len(verts), :] for i, verts in enumerate(scan_mesh_verts)]
        
        # Add scan skinning weights to batch
        batch['scan_skinning_weights'] = scan_w


        # build pytorch3d cameras
        R, T, K = batch['R'], batch['T'], batch['K']
        
        # print(T.shape)
        # print(R.shape)
        # print(K.shape)
        R = R.view(-1, 3, 3).float()
        T = T.view(-1, 3).float()
        K = K.view(-1, 4, 4).float()
        
        # Scale K matrix for 2x downsampling
        # For 2x downsampling, scale focal lengths and principal point by 0.5
        # scale_factor = 224/940
        scale_factor = 1
        K_scaled = K.clone()
        K_scaled[:, 0, 0] *= scale_factor  # fx
        K_scaled[:, 1, 1] *= scale_factor  # fy
        K_scaled[:, 0, 2] *= scale_factor  # cx
        K_scaled[:, 1, 2] *= scale_factor  # cy
        
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            K=K_scaled,
            image_size=[(1280*scale_factor, 940*scale_factor)],  # Scaled down from (1280, 940)
            device=self.device,
            in_ndc=False
        )
        self.feature_renderer._set_cameras(cameras)
        

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


        vp = [v for sublist in batch['scan_mesh_verts'] for v in sublist]
        
        vp_ptcld = Pointclouds(points=vp)
        batch['vp_ptcld'] = vp_ptcld



        # Visualize rendered maps in a 2x4 grid
        # to_vis = torch.argmax(w_maps, dim=-1).cpu().detach().numpy()
        # import matplotlib.pyplot as plt
        
        # fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        # for i in range(2):
        #     for j in range(4):
        #         idx = i * 4 + j
        #         axes[i,j].imshow(to_vis[i, j])
        #         axes[i,j].axis('off')
        
        # plt.tight_layout()
        # plt.savefig('rendered_maps.png')
        # plt.close()

        # import ipdb; ipdb.set_trace()

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
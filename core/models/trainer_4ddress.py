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
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points

from core.configs import paths 
from core.models.smpl import SMPL
from core.models.cch import CCH 
from core.losses.cch_loss import CCHLoss
from core.losses.cch_metrics import CCHMetrics
from core.utils.visualiser import Visualiser
from core.utils.feature_renderer import FeatureRenderer
from core.models.sapiens_wrapper import SapiensWrapper
from core.configs.model_size_cfg import MODEL_CONFIGS



class CCHTrainer(pl.LightningModule):
    def __init__(self, 
                 cfg, 
                 dev=False, 
                 vis_save_dir=None,
                 plot=False):
        
        super().__init__()
        self.save_scenepic = True 
        self.dev = dev
        self.cfg = cfg
        self.use_sapiens = cfg.MODEL.USE_SAPIENS
        self.normalise = cfg.DATA.NORMALISE
        self.vis_frequency = cfg.VISUALISE_FREQUENCY if not dev else 5
        self.image_size = MODEL_CONFIGS[cfg.MODEL.SIZE]['img_size']
        self.plot = plot

        # self.feature_renderer = FeatureRenderer(image_size=(224, 224))
        self.feature_renderer = FeatureRenderer(image_size=(512, 376)) # fixed according to 4DDress size 1280 * 940, interpolated later to img_size * img_size for CCH input 

        self.smpl_male = SMPL(
            model_path=paths.SMPL,
            num_betas=10,
            gender='male'
        )
        self.smpl_female = SMPL(
            model_path=paths.SMPL,  
            num_betas=10,
            gender='female'
        )
        for param in self.smpl_male.parameters():
            param.requires_grad = False
        for param in self.smpl_female.parameters():
            param.requires_grad = False

        self.parents = self.smpl_male.parents

        if self.use_sapiens:
            model_cfg = MODEL_CONFIGS[cfg.MODEL.SIZE]
            self.sapiens = SapiensWrapper()
            for param in self.sapiens.parameters():
                param.requires_grad = False
        else:
            self.sapiens = None

        self.model = CCH(
            cfg=cfg,
            smpl_male=self.smpl_male,
            smpl_female=self.smpl_female,
            sapiens=self.sapiens
        )

        self.criterion = CCHLoss(cfg)
        self.metrics = CCHMetrics(cfg)
        self.visualiser = Visualiser(save_dir=vis_save_dir, cfg=cfg)

        self.save_hyperparameters(ignore=['smpl_model'])

        self.first_batch = None


    def forward(self, batch):
        return self.model(batch)
    
    def on_train_epoch_start(self):
        self.visualiser.set_global_rank(self.global_rank)

    def training_step(self, batch, batch_idx, split='train'):
        if self.first_batch is None:
            self.first_batch = batch
        if self.dev:
            batch = self.first_batch

        batch = self._process_inputs(batch, batch_idx, normalise=self.normalise)


        # import matplotlib.pyplot as plt
        
        # B, N = batch['imgs'].shape[:2]
        # fig, axes = plt.subplots(B, N, figsize=(4*N, 4*B))
        
        # # Handle single batch/frame case
        # if B == 1 and N == 1:
        #     axes = np.array([[axes]])
        # elif B == 1:
        #     axes = axes[None, :]
        # elif N == 1: 
        #     axes = axes[:, None]
            
        # for b in range(B):
        #     for n in range(N):
        #         img = batch['imgs'][b,n].detach().cpu().permute(1,2,0).numpy()
        #         axes[b,n].imshow(img)
        #         # axes[b,n].axis('off')
                
        # plt.tight_layout()
        # plt.show()
        # plt.savefig('test_input_images.png')
        # import ipdb; ipdb.set_trace()
    
        preds = self(
            batch
        )

        loss, loss_dict = self.criterion(preds, batch)

        metrics = self.metrics(preds, batch)

        self._log_metrics_and_visualise(loss, loss_dict, metrics, split, preds, batch, self.global_step)

        # print(loss_dict)
        # import ipdb; ipdb.set_trace()
        
        return loss 
    

    def _log_metrics_and_visualise(self, loss, loss_dict, metrics, split, preds, batch, global_step):
        self.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, rank_zero_only=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

        # if (global_step % self.vis_frequency == 0 and global_step > 0) or (global_step == 1):
        # self.visualiser.visualise(preds, batch) 

        if self.dev or self.plot:
            self.visualiser.visualise(preds, batch)
            if self.dev:
                import ipdb; ipdb.set_trace()
    

    @torch.no_grad()
    def _process_inputs(self, batch, batch_idx, normalise=False):

        B, N = batch['imgs'].shape[:2]

        # ----------------------- get T joints -----------------------
        smpl_T_joints_list = []
        smpl_T_vertices_list = []
        smpl_vertices_list = []
        for i in range(B):
            if batch['gender'][i] == 'male':
                smpl_model = self.smpl_male
            else:
                smpl_model = self.smpl_female

            smpl_T_output = smpl_model(
                betas=batch['betas'][i].view(N, 10),
                body_pose = torch.zeros((N, 69)).to(self.device),
                global_orient = torch.zeros((N, 3)).to(self.device),
                transl = torch.zeros((N, 3)).to(self.device)
            )
            smpl_T_joints = smpl_T_output.joints[:, :24]
            smpl_T_vertices = smpl_T_output.vertices
            smpl_T_joints_list.append(smpl_T_joints)
            smpl_T_vertices_list.append(smpl_T_vertices)

            smpl_output = smpl_model(
                betas=batch['betas'][i].view(N, 10),
                body_pose = batch['body_pose'][i].view(N, 69),
                global_orient = batch['global_orient'][i].view(N, 3),
                transl = batch['transl'][i].view(N, 3)
            )
            smpl_vertices = smpl_output.vertices
            smpl_vertices_list.append(smpl_vertices)

        smpl_T_joints = torch.stack(smpl_T_joints_list, dim=0)
        smpl_T_vertices = torch.stack(smpl_T_vertices_list, dim=0)
        smpl_vertices = torch.stack(smpl_vertices_list, dim=0)

        smpl_skinning_weights = (self.smpl_male.lbs_weights)[None, None].repeat(B, N, 1, 1)

        smpl_faces = torch.tensor(self.smpl_male.faces, dtype=torch.int32)

        batch['smpl_T_joints'] = smpl_T_joints


        scan_meshes = batch['scan_mesh']
        scan_mesh_verts = [v for sublist in batch['scan_mesh_verts'] for v in sublist]
        scan_mesh_faces = [f for sublist in batch['scan_mesh_faces'] for f in sublist]
        scan_mesh_verts_centered = [v for sublist in batch['scan_mesh_verts_centered'] for v in sublist]



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




        # Render skinning weight pointmaps
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



        # Render SMPL pointmaps
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
        mask = torch.nn.functional.interpolate(mask.permute(0,3,1,2), size=(224,224), mode='nearest')
        mask = rearrange(mask, '(b n) c h w -> b n h w c', b=B, n=N)
        batch['smpl_mask'] = mask.squeeze(-1)



        # vp = [v for sublist in batch['scan_mesh_verts_centered'] for v in sublist]
        # vp_ptcld = Pointclouds(points=vp)
        # batch['vp_ptcld'] = vp_ptcld


        # Sample from Vp
        scan_mesh_centered = Meshes(
            verts=scan_mesh_verts_centered,
            faces=scan_mesh_faces,
        )
        vp = sample_points_from_meshes(scan_mesh_centered, 48000)
        vp_ptcld = Pointclouds(points=vp)
        batch['vp_ptcld'] = vp_ptcld


        # Sample from Template Mesh
        template_mesh = batch['template_mesh']
        template_mesh_verts = [mesh.vertices for mesh in template_mesh]
        template_mesh_faces = [mesh.faces for mesh in template_mesh]


        # Align Template Mesh with SMPL
        smpl_T_vertices_midpoint = (torch.max(smpl_T_vertices[..., 1], dim=-1)[0] + torch.min(smpl_T_vertices[..., 1], dim=-1)[0]) / 2
        template_mesh_verts_midpoint = torch.stack([(torch.max(torch.tensor(verts[:, 1], device=self.device)) + torch.min(torch.tensor(verts[:, 1], device=self.device))) / 2 for verts in template_mesh_verts])
        offset = (smpl_T_vertices_midpoint[:, 0] - template_mesh_verts_midpoint)

        template_mesh_verts = [torch.tensor(verts, device=self.device, dtype=torch.float32) for i, verts in enumerate(template_mesh_verts)]
        template_mesh_verts[0][:, 1] += offset[0]


        template_mesh_pytorch3d = Meshes(
            verts=template_mesh_verts,
            faces=[torch.tensor(f, device=self.device, dtype=torch.int32) for f in template_mesh_faces]
        )

        batch['template_mesh_verts'] = sample_points_from_meshes(template_mesh_pytorch3d, 48000)




        # # Create 2D plot comparing SMPL and template vertices
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))

        # # Plot SMPL vertices
        # plt.subplot(121)
        # plt.scatter(smpl_T_vertices[0, 0, :, 0].cpu().numpy(), 
        #            smpl_T_vertices[0, 0, :, 1].cpu().numpy(),
        #            c='blue', alpha=0.5, s=1)
        # plt.title('SMPL T-Pose Vertices')
        # plt.axis('equal')

        # # Plot template vertices 
        # plt.subplot(122)
        # plt.scatter(batch['template_mesh_verts'][0][:, 0].cpu().numpy(),
        #            batch['template_mesh_verts'][0][:, 1].cpu().numpy(), 
        #            c='red', alpha=0.5, s=1)
        # plt.title('Template Mesh Vertices')
        # plt.axis('equal')

        # # Get min/max ranges for both plots
        # smpl_x_min = smpl_T_vertices[0, 0, :, 0].cpu().numpy().min()
        # smpl_x_max = smpl_T_vertices[0, 0, :, 0].cpu().numpy().max()
        # smpl_y_min = smpl_T_vertices[0, 0, :, 1].cpu().numpy().min() 
        # smpl_y_max = smpl_T_vertices[0, 0, :, 1].cpu().numpy().max()

        # template_x_min = batch['template_mesh_verts'][0][:, 0].cpu().numpy().min()
        # template_x_max = batch['template_mesh_verts'][0][:, 0].cpu().numpy().max()
        # template_y_min = batch['template_mesh_verts'][0][:, 1].cpu().numpy().min()
        # template_y_max = batch['template_mesh_verts'][0][:, 1].cpu().numpy().max()

        # # Set same range for both plots
        # x_min = min(smpl_x_min, template_x_min)
        # x_max = max(smpl_x_max, template_x_max)
        # y_min = min(smpl_y_min, template_y_min)
        # y_max = max(smpl_y_max, template_y_max)

        # plt.subplot(121).set_xlim(x_min, x_max)
        # plt.subplot(121).set_ylim(y_min, y_max)
        # plt.subplot(122).set_xlim(x_min, x_max)
        # plt.subplot(122).set_ylim(y_min, y_max)

        # plt.tight_layout()
        # plt.savefig(f'vertex_comparison.png')
        # plt.close()

        # import ipdb; ipdb.set_trace()




        # self._test_smpl_scan_alignment(smpl_vertices, scan_mesh_verts)
        # self._test_render(vc_maps, masks=mask, name='vc_maps')
        # self._test_render(w_maps, name='w_maps')
        # self._test_sampling(scan_mesh_centered, vp_ptcld)

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
            mask = masks.cpu().detach().numpy().squeeze(-1)
            to_vis[~mask.astype(bool)] = 0
            norm_min, norm_max = to_vis.min(), to_vis.max()
            to_vis = (to_vis - norm_min) / (norm_max - norm_min) 
            to_vis[~mask.astype(bool)] = 1

        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(1):
            for j in range(4):
                idx = i * 4 + j
                axes[i,j].imshow(to_vis[i, j])
                # axes[i,j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{name}_rendered_maps.png')
        plt.close()

        import ipdb; ipdb.set_trace()
        return None

    def _test_sampling(self, scan_mesh_centered, vp_ptcld):

        # Visualize each scan mesh in scan_mesh_centered using pytorch3d plotly_vis
        from pytorch3d.vis.plotly_vis import plot_scene
        # Create a scene dict with each mesh as a separate subplot
        scene_dict = {}
        for i in range(len(scan_mesh_centered)):
            scene_dict[f"scan_mesh_{i}"] = {
                "scan": Meshes(
                    verts=[scan_mesh_centered.verts_list()[i]], 
                    faces=[scan_mesh_centered.faces_list()[i]]
                )
            }
            
        # Plot all meshes in separate subplots
        fig = plot_scene(
            scene_dict,
            ncols=min(4, len(scan_mesh_centered)), # Max 4 columns
            camera_scale=0.5,
            viewpoint_cameras=None
        )
        fig.write_image("scan_meshes.png")

        # Visualize point clouds using pytorch3d plotly_vis
        scene_dict = {}
        for i in range(len(vp_ptcld)):
            scene_dict[f"pointcloud_{i}"] = {
                "points": vp_ptcld[i]
            }
            
        # Plot all point clouds in separate subplots
        fig = plot_scene(
            scene_dict,
            ncols=min(4, len(vp_ptcld)), # Max 4 columns
            camera_scale=0.5,
            viewpoint_cameras=None
        )
        fig.write_image("pointclouds.png")
        import ipdb; ipdb.set_trace()
import torch
import smplx
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
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
# from core.models.smpl import SMPL
from core.models.cch import CCH 
from core.losses.cch_loss import CCHLoss
from core.losses.cch_metrics import CCHMetrics
from core.utils.visualiser import Visualiser
from core.utils.feature_renderer import FeatureRenderer
from core.models.sapiens_wrapper import SapiensWrapper
from core.configs.model_size_cfg import MODEL_CONFIGS
from core.utils.general_lbs import general_lbs
from core.utils.smpl_utils import get_smplx_full_pose



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
        self.num_samples = cfg.DATA.NUM_SAMPLES
        self.B = cfg.TRAIN.BATCH_SIZE
        self.use_sapiens = cfg.MODEL.USE_SAPIENS
        self.normalise = cfg.DATA.NORMALISE
        self.vis_frequency = cfg.VISUALISE_FREQUENCY if not dev else 5
        self.image_size = cfg.DATA.IMAGE_SIZE
        self.plot = plot
        self.body_model = cfg.MODEL.BODY_MODEL
        self.d4dress_probability = cfg.DATA.D4DRESS_PROBABILITY
        self.num_joints = 24 if self.body_model=='smpl' else 55
        self.num_smpl_vertices = 6890 if self.body_model=='smpl' else 10475
        self.freeze_canonical_epochs = getattr(cfg.TRAIN, 'WARMUP_EPOCHS', 0)


        self.feature_renderer = FeatureRenderer(image_size=(512, 384))#(512, 384))#image_size=(256, 192)) 
        # self.thuman_renderer = FeatureRenderer(image_size=(256, 256))

        self.smpl_male = smplx.create(
            model_type=self.body_model,
            model_path="model_files/",
            num_betas=10,
            gender='male',
            num_pca_comps=12,
        )
        self.smpl_female = smplx.create(
            model_type=self.body_model,
            model_path="model_files/",
            num_betas=10,
            gender='female',
            num_pca_comps=12,
        )
        self.smpl_neutral = smplx.create(
            model_type=self.body_model,
            model_path="model_files/",
            num_betas=10,
            gender='neutral',
            num_pca_comps=12,
        )
        for param in self.smpl_male.parameters():
            param.requires_grad = False
        for param in self.smpl_female.parameters():
            param.requires_grad = False
        for param in self.smpl_neutral.parameters():
            param.requires_grad = False

        self.parents = self.smpl_male.parents
        
        self.model = CCH(
            cfg=cfg,
            smpl_male=self.smpl_male,
            smpl_female=self.smpl_female,
        )
        # self._freeze_canonical_modules()

        self.criterion = CCHLoss(cfg)
        self.metrics = CCHMetrics(cfg)
        self.visualiser = Visualiser(save_dir=vis_save_dir, cfg=cfg)

        self.save_hyperparameters(ignore=['smpl_male', 'smpl_female'])

        self.first_batch = None


    def forward(self, batch):
        return self.model(batch)
    
    def on_train_epoch_start(self):
        self.visualiser.set_global_rank(self.global_rank)
        
        # # Alternate between canonical and PBS training based on epoch
        # current_epoch = self.current_epoch
        # if current_epoch % 2 == 1:  # Odd epochs: train canonical only
        #     self._set_train_vc()
        # else:  # Odd epochs: train PBS only
        #     self._set_train_vp()
        
        # # Update optimizer parameters after changing requires_grad status
        # self._update_optimizer_parameters()


    def training_step(self, batch, batch_idx, split='train'):
        if self.first_batch is None:
            self.first_batch = batch
        if self.dev:
            batch = self.first_batch

        batch = batch[0] if np.random.rand() > self.d4dress_probability else batch[1]
        if batch['dataset'][0] == '4DDress':
            batch = self.process_4ddress(batch, batch_idx, normalise=self.normalise)
        elif batch['dataset'][0] == 'THuman':
            batch = self.process_thuman(batch)

        preds = self(batch)

        loss, loss_dict = self.criterion(preds, batch, dataset_name=batch['dataset'][0])

        metrics = self.metrics(preds, batch)

        # self.visualiser.visualise_debug_loss(loss_dict)

        # loss_dict.pop('debug_loss_pred2gt_conf')
        # loss_dict.pop('debug_loss_pred2gt')
        # loss_dict.pop('debug_loss_gt2pred')
        self._log_metrics_and_visualise(loss, loss_dict, metrics, split, preds, batch, batch_idx)

        
        # for k, v in loss_dict.items():
        #     print(f"{k}: {v.item():.2f}", end='; ')
        # print('')
        # import ipdb; ipdb.set_trace()
        
        return loss 



    def _log_metrics_and_visualise(self, loss, loss_dict, metrics, split, preds, batch, batch_idx=None):
        if split == 'train' or split == 'test':
            on_step = True
        else:
            on_step = False
            
        for key in list(loss_dict.keys()):
            loss_dict[f'{split}_{key}'] = loss_dict.pop(key)
        for key in list(metrics.keys()):
            metrics[f'{split}_{key}'] = metrics.pop(key)


        
        self.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True)
        if f'{split}_vc_cfd' in metrics:
            self.log(f'{split}_vc_cfd', metrics.pop(f'{split}_vc_cfd'), 
                     on_step=on_step, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=self.B)
        if f'{split}_vp_init_cfd' in metrics:
            self.log(f'{split}_vp_init_cfd', metrics.pop(f'{split}_vp_init_cfd'), 
                     on_step=on_step, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=self.B)
        if f'{split}_vp_cfd' in metrics:
            self.log(f'{split}_vp_cfd', metrics.pop(f'{split}_vp_cfd'), 
                     on_step=on_step, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=self.B)
        self.log_dict(loss_dict, on_step=on_step, on_epoch=True, prog_bar=False, rank_zero_only=True, sync_dist=True, batch_size=self.B)
        self.log_dict(metrics, on_step=on_step, on_epoch=True, prog_bar=False, rank_zero_only=True, sync_dist=True, batch_size=self.B)

        if self.dev or self.plot:
            # Synchronize CUDA on ALL ranks before visualization to ensure DDP processes stay in sync
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.visualiser.visualise(preds, batch, split=split, epoch=self.current_epoch)
            # Synchronize CUDA on ALL ranks after visualization to ensure DDP processes stay in sync
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if self.dev:
                for k, v in loss_dict.items():
                    print(f"{k}: {v.item():.2f}", end='; ')
                print('')
                # import ipdb; ipdb.set_trace()

        else:
            should_vis = False
            global_step = self.global_step
            if split in ('train', 'test'):
                should_vis = ((global_step % self.vis_frequency == 0 and global_step > 0) or (global_step == 1))
            elif split == 'val':
                # Visualise every N validation batches based on vis_frequency
                should_vis = (batch_idx == 4)

            if should_vis:
                # Synchronize CUDA on ALL ranks before visualization to ensure DDP processes stay in sync
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.visualiser.visualise(preds, batch, split=split, epoch=self.current_epoch)
                # Synchronize CUDA on ALL ranks after visualization to ensure DDP processes stay in sync
                if torch.cuda.is_available():
                    torch.cuda.synchronize() 


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Handle both 4DDress and THuman validation datasets
        # dataloader_idx: 0 = THuman, 1 = 4DDress (matching order in val_dataloader return)
        if dataloader_idx == 0:
            dataset_name = 'THuman'
            batch = self.process_thuman(batch)
        elif dataloader_idx == 1:
            dataset_name = '4DDress'
            batch = self.process_4ddress(batch, batch_idx, normalise=self.normalise)
        else:
            # Fallback: try to determine from batch structure
            if isinstance(batch, dict) and 'dataset' in batch:
                dataset_name = batch['dataset'][0] if isinstance(batch['dataset'], list) else batch['dataset']
            else:
                dataset_name = '4DDress' if 'vc_maps' in batch else 'THuman'
            
            if dataset_name == '4DDress':
                batch = self.process_4ddress(batch, batch_idx, normalise=self.normalise)
            elif dataset_name == 'THuman':
                batch = self.process_thuman(batch)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        
        preds = self(batch)
        loss, loss_dict = self.criterion(preds, batch, dataset_name=dataset_name)
        metrics = self.metrics(preds, batch)
        self._log_metrics_and_visualise(loss, loss_dict, metrics, 'val', preds, batch, batch_idx)
        # with torch.no_grad():
        #     loss = self.training_step(batch, batch_idx, split='val')
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)
    

    
    @torch.no_grad()
    def process_4ddress(self, batch, batch_idx, normalise=False):
        with torch.autocast(enabled=False, device_type='cuda'):

            B, K = batch['imgs'].shape[:2]
            assert K == 5
            N = 4 

            batch['imgs'] = batch['imgs'] * batch['masks'].unsqueeze(2)

            # ----------------------- SMPL / SMPLx -----------------------
            smpl_T_joints_list = []
            smpl_T_vertices_list = []
            smpl_vertices_list = []
            smpl_skinning_weights_list = []
            smpl_full_pose_list = []
            for i in range(B):
                if batch['gender'][i] == 'male':
                    smpl_model = self.smpl_male
                else:
                    smpl_model = self.smpl_female


                if self.body_model == 'smplx':
                    smpl_T_output = smpl_model(
                        betas=batch['betas'][i].view(K, -1),
                        body_pose = torch.zeros_like(batch['body_pose'][i].view(K, -1)),
                        global_orient = torch.zeros_like(batch['global_orient'][i].view(K, -1)),
                        transl = torch.zeros_like(batch['transl'][i].view(K, -1)),
                        left_hand_pose = torch.zeros_like(batch['left_hand_pose'][i].view(K, -1)),
                        right_hand_pose = torch.zeros_like(batch['right_hand_pose'][i].view(K, -1)),
                        expression = torch.zeros_like(batch['expression'][i].view(K, -1)),
                        jaw_pose = torch.zeros_like(batch['jaw_pose'][i].view(K, -1)),
                        leye_pose = torch.zeros_like(batch['leye_pose'][i].view(K, -1)),
                        reye_pose = torch.zeros_like(batch['reye_pose'][i].view(K, -1)),
                    )   
                    smpl_output = smpl_model(
                        betas=batch['betas'][i].view(K, -1),
                        global_orient = batch['global_orient'][i].view(K, -1),
                        body_pose = batch['body_pose'][i].view(K, -1),
                        left_hand_pose = batch['left_hand_pose'][i].view(K, -1),
                        right_hand_pose = batch['right_hand_pose'][i].view(K, -1),
                        transl = batch['transl'][i].view(K, -1),
                        expression = batch['expression'][i].view(K, -1),
                        jaw_pose = batch['jaw_pose'][i].view(K, -1),
                        leye_pose = batch['leye_pose'][i].view(K, -1),
                        reye_pose = batch['reye_pose'][i].view(K, -1),
                        return_full_pose=True,
                    )
                elif self.body_model == 'smpl':
                    smpl_T_output = smpl_model(
                        betas=batch['betas'][i].view(K, -1),
                        body_pose = torch.zeros_like(batch['body_pose'][i].view(K, -1)),
                        global_orient = torch.zeros_like(batch['global_orient'][i].view(K, -1)),
                        transl = torch.zeros_like(batch['transl'][i].view(K, -1)),
                    )
                    smpl_output = smpl_model(
                        betas=batch['betas'][i].view(K, -1),
                        body_pose = batch['body_pose'][i].view(K, -1),
                        global_orient = batch['global_orient'][i].view(K, -1),
                        transl = batch['transl'][i].view(K, -1),
                        return_full_pose=True,
                    )
                else:
                    raise ValueError(f"Body model {self.body_model} not supported")
                
                smpl_T_joints = smpl_T_output.joints[:, :self.num_joints]
                smpl_T_vertices = smpl_T_output.vertices

                smpl_T_joints_list.append(smpl_T_joints)
                smpl_T_vertices_list.append(smpl_T_vertices)
                smpl_vertices_list.append(smpl_output.vertices)
                smpl_skinning_weights_list.append(smpl_model.lbs_weights)
                smpl_full_pose_list.append(smpl_output.full_pose)
                
            smpl_T_joints = torch.stack(smpl_T_joints_list, dim=0)#.repeat(1, K, 1, 1)
            smpl_T_vertices = torch.stack(smpl_T_vertices_list, dim=0)
            smpl_vertices = torch.stack(smpl_vertices_list, dim=0)
            smpl_skinning_weights = torch.stack(smpl_skinning_weights_list, dim=0)[:, None].repeat(1, K, 1, 1)
            smpl_full_pose = torch.stack(smpl_full_pose_list, dim=0)

            batch['pose'] = smpl_full_pose
            batch['smpl_T_joints'] = smpl_T_joints

            # ----------------------- Scan Mesh -----------------------
            scan_meshes = batch['scan_mesh']
            scan_mesh_verts = [v for sublist in batch['scan_mesh_verts'] for v in sublist]
            scan_mesh_faces = [f for sublist in batch['scan_mesh_faces'] for f in sublist]
            scan_mesh_verts_centered = [v for sublist in batch['scan_mesh_verts_centered'] for v in sublist]

            dists, idx = self.knn_ptcld(
                Pointclouds(points=scan_mesh_verts), 
                smpl_vertices.view(-1, smpl_vertices.shape[-2], 3), 
                K=1
            )
            smpl_weights_flat = smpl_skinning_weights.view(-1, self.num_smpl_vertices, self.num_joints)
            idx_expanded = idx.repeat(1, 1, self.num_joints)
            scan_w_tensor = torch.gather(smpl_weights_flat, dim=1, index=idx_expanded)
            scan_w = [scan_w_tensor[i, :len(verts), :] for i, verts in enumerate(scan_mesh_verts)]
            batch['scan_skinning_weights'] = scan_w




            # ----------------------- Render -----------------------
            # build pytorch3d cameras
            R, T, cam_K = batch['R'], batch['T'], batch['K']
            R = R.view(-1, 3, 3).float()
            T = T.view(-1, 3).float()
            cam_K = cam_K.view(-1, 4, 4).float()

            cameras = PerspectiveCameras(
                R=R, T=T, K=cam_K,
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
            )

            renderer_output = self.feature_renderer(pytorch3d_mesh)
            w_maps = renderer_output['maps']
            # visible_faces = renderer_output['visible_faces']

            _, H, W, _ = w_maps.shape
            target_size = W 
            crop_amount = (H - target_size) // 2  
            w_maps = w_maps[:, crop_amount:H-crop_amount, :, :]
            w_maps = torch.nn.functional.interpolate(
                w_maps.permute(0,3,1,2), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
            )
            w_maps = rearrange(w_maps, '(b k) j h w -> b k h w j', b=B, k=K)
            
            batch['smpl_w_maps'] = w_maps


            template_mesh = batch['template_mesh']
            template_mesh_verts = [torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32) for mesh in template_mesh]
            template_mesh_faces = [torch.tensor(mesh.faces, device=self.device, dtype=torch.long) for mesh in template_mesh]


            template_full_mesh = batch['template_full_mesh']
            template_full_lbs_weights = batch['template_full_lbs_weights']
            
            template_full_mesh_verts = [torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32) for mesh in template_full_mesh]
            template_full_mesh_faces = [torch.tensor(mesh.faces, device=self.device, dtype=torch.long) for mesh in template_full_mesh]

            template_full_mesh_verts_posed_list = []
            template_full_mesh_faces_expanded_list = [faces[None].repeat(K, 1, 1) for faces in template_full_mesh_faces]
            template_full_mesh_verts_expanded_list = [verts[None].repeat(K, 1, 1) for verts in template_full_mesh_verts]

            for b in range(B):
                template_full_mesh_verts_posed, _ = general_lbs(
                    pose=batch['pose'][b],
                    J=batch['smpl_T_joints'][b],
                    vc=template_full_mesh_verts[b][None].repeat(K, 1, 1),
                    lbs_weights=template_full_lbs_weights[b][None].repeat(K, 1, 1),
                    parents=self.smpl_male.parents,
                )
                template_full_mesh_verts_posed += batch['transl'][b][:, None, :]
                template_full_mesh_verts_posed_list.append(template_full_mesh_verts_posed)


            # Flatten list of B items of shape (K,N,3) into B*K items of shape (N,3)
            template_full_mesh_verts_posed_list = [
                verts[k] for verts in template_full_mesh_verts_posed_list 
                for k in range(K) 
            ]
            template_full_mesh_faces_expanded_list = [
                faces[k] for faces in template_full_mesh_faces_expanded_list 
                for k in range(K) 
            ]
            template_full_mesh_verts_expanded_list = [
                verts[k] for verts in template_full_mesh_verts_expanded_list 
                for k in range(K) 
            ]

            if self.cfg.DATA.NORMALISE:
                normalise_to_height = 1.7 
                smpl_T_height = (smpl_T_vertices[..., 1].max(dim=-1).values - smpl_T_vertices[..., 1].min(dim=-1).values).flatten() # b * k
                template_full_mesh_verts_expanded_list = [
                    verts * (normalise_to_height / smpl_T_height[i]) for i, verts in enumerate(template_full_mesh_verts_expanded_list)
                ]


            template_full_posed_pytorch3d_mesh = Meshes(
                verts=template_full_mesh_verts_posed_list,
                faces=template_full_mesh_faces_expanded_list,
                textures=TexturesVertex(verts_features=template_full_mesh_verts_expanded_list)
            )


            ret = self.feature_renderer(template_full_posed_pytorch3d_mesh)
            vc_maps = ret['maps']
            mask = ret['mask'].unsqueeze(-1)
            _, H, W, _ = vc_maps.shape
            target_size = W 
            crop_amount = (H - target_size) // 2  
            vc_maps = vc_maps[:, crop_amount:H-crop_amount, :, :]
            vc_maps = torch.nn.functional.interpolate(
                vc_maps.permute(0,3,1,2), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
            )
            vc_maps = rearrange(vc_maps, '(b k) c h w -> b k h w c', b=B, k=K)
            batch['vc_maps'] = vc_maps

            mask = mask[:, crop_amount:H-crop_amount, :, :]
            mask = torch.nn.functional.interpolate(
                mask.permute(0,3,1,2), size=(self.image_size, self.image_size), mode='nearest'
            )
            mask = rearrange(mask, '(b k) c h w -> b k h w c', b=B, k=K)
            batch['smpl_mask'] = mask.squeeze(-1)

            
            template_mesh = batch['template_mesh']
            template_mesh_verts = [torch.tensor(mesh.vertices, device=self.device, dtype=torch.float32) for mesh in template_mesh]
            template_mesh_faces = [torch.tensor(mesh.faces, device=self.device, dtype=torch.long) for mesh in template_mesh]


            # -----------------Normalise after render -----------------
            if self.cfg.DATA.NORMALISE:
                normalise_to_height = 1.7 
                smpl_T_height = (smpl_T_vertices[..., 1].max(dim=-1).values - smpl_T_vertices[..., 1].min(dim=-1).values) # b * k
                smpl_T_joints = smpl_T_joints[:, :self.num_joints] * (normalise_to_height / smpl_T_height)[:, :, None, None] # b k j 3 * b k 1 1
                smpl_T_height = smpl_T_height.flatten()
                scan_mesh_verts_centered = [verts * (normalise_to_height / smpl_T_height[i]) for i, verts in enumerate(scan_mesh_verts_centered)]
                template_mesh_verts = [verts * (normalise_to_height / smpl_T_height[i]) for i, verts in enumerate(template_mesh_verts)]

            batch['smpl_T_joints'] = smpl_T_joints



            # ----------------------- Sampling -----------------------
            # Sample from Vp
            scan_mesh_centered = Meshes(
                verts=scan_mesh_verts_centered,
                faces=scan_mesh_faces,
            )
            vp = sample_points_from_meshes(scan_mesh_centered, self.num_samples)
            vp_ptcld = Pointclouds(points=vp)
            batch['vp_ptcld'] = vp_ptcld
            batch['vp'] = vp



            template_mesh_pytorch3d = Meshes(
                verts=template_mesh_verts,
                faces=template_mesh_faces
            )

            batch['template_mesh_verts'] = sample_points_from_meshes(template_mesh_pytorch3d, self.num_samples)


            # self._test_smpl_scan_alignment(smpl_vertices, scan_mesh_verts)
            # self._test_render(vc_maps, masks=mask, name='vc_maps')
            # self._test_render(w_maps, name='w_maps')
            # self._test_sampling(scan_mesh_centered, vp_ptcld)

        return batch 
    

    # will be used in trainer 
    @torch.no_grad()
    def process_thuman(self, batch):
        with torch.autocast(enabled=False, device_type='cuda'):
            B, K = batch['imgs'].shape[:2]
            N = 4

            smpl_model = self.smpl_neutral # neutral is used to fit THuman 
            smpl_skinning_weights = self.smpl_male.lbs_weights

            smplx_params = [params for sublist in batch['smplx_param'] for params in sublist]
            smplx_params = {
                k: torch.stack([torch.tensor(d[k], device=self.device, dtype=torch.float32) for d in smplx_params]).flatten(0, 1)
                for k in smplx_params[0].keys()
            }
            smplx_params.pop('transl')

            smplx_T_params = {
                k: torch.zeros_like(smplx_params[k])
                for k in smplx_params.keys()
            }
            smplx_T_params['betas'] = smplx_params['betas']

            smpl_output = smpl_model(return_full_pose=True, **smplx_params)
            smpl_T_output = smpl_model(return_full_pose=True, **smplx_T_params)


            scan_verts = [verts for sublist in batch['scan_verts'] for verts in sublist]
            scan_faces = [faces for sublist in batch['scan_faces'] for faces in sublist]




            # ----------------------- Render -----------------------
            batch['smpl_w_maps'] = rearrange(batch['smpl_w_maps'], 'b k c h w -> b k h w c')
            batch['vc_smpl_maps'] = rearrange(batch['vc_smpl_maps'], 'b k c h w -> b k h w c')

            # cam_R, cam_T = batch['cam_R'], batch['cam_T']

            # cameras = PerspectiveCameras(
            #     R=cam_R.flatten(0, 1), 
            #     T=cam_T.flatten(0, 1), 
            #     focal_length=724.0773/2,
            #     principal_point=[(128, 128),],
            #     image_size=[(256, 256),],
            #     device=self.device,
            #     in_ndc=False
            # )
            # self.thuman_renderer._set_cameras(cameras)

            # dists, idx = self.knn_ptcld(
            #     Pointclouds(points=scan_verts), 
            #     smpl_output.vertices.view(-1, smpl_output.vertices.shape[-2], 3), 
            #     K=1
            # )
            # smpl_weights_flat = smpl_skinning_weights.expand(B*K, -1, -1)
            # idx_expanded = idx.repeat(1, 1, self.num_joints)
            # scan_w_tensor = torch.gather(smpl_weights_flat, dim=1, index=idx_expanded)
            # scan_w = [scan_w_tensor[i, :len(verts), :] for i, verts in enumerate(scan_verts)]
            # batch['scan_skinning_weights'] = scan_w


            # # Render skinning weight pointmaps
            # pytorch3d_mesh = Meshes(
            #     verts=scan_verts,
            #     faces=scan_faces,
            #     textures=TexturesVertex(verts_features=scan_w)
            # )

            # renderer_output = self.thuman_renderer(pytorch3d_mesh)
            # w_maps = renderer_output['maps']

            # _, H, W, _ = w_maps.shape
            # target_size = W 
            # crop_amount = (H - target_size) // 2  
            # w_maps = w_maps[:, crop_amount:H-crop_amount, :, :]
            # w_maps = torch.nn.functional.interpolate(
            #     w_maps.permute(0,3,1,2), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
            # )
            # w_maps = rearrange(w_maps, '(b k) j h w -> b k h w j', b=B, k=K)
            
            # batch['smpl_w_maps'] = w_maps

            # if self.cfg.DATA.NORMALISE:
            #     normalise_to_height = 1.7 
            #     smpl_T_height = smpl_T_output.vertices[:, :, 1].max(dim=-1).values - smpl_T_output.vertices[:, :, 1].min(dim=-1).values
            #     smpl_T_vertices_normalised = smpl_T_output.vertices * (normalise_to_height / smpl_T_height)[:, None, None]

            # pytorch3d_smpl_mesh = Meshes(
            #     verts=smpl_output.vertices,
            #     faces=torch.tensor(smpl_model.faces, device=self.device).expand(B*K, -1, -1),
            #     textures=TexturesVertex(verts_features=smpl_T_vertices_normalised)
            # )

            # renderer_output = self.thuman_renderer(pytorch3d_smpl_mesh)
            # vc_maps = renderer_output['maps']
            # vc_mask = renderer_output['mask'].unsqueeze(-1)
            # vc_maps = torch.nn.functional.interpolate(
            #     vc_maps.permute(0,3,1,2), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
            # )
            # vc_maps = rearrange(vc_maps, '(b k) j h w -> b k h w j', b=B, k=K)

            # vc_mask = torch.nn.functional.interpolate(
            #     vc_mask.permute(0,3,1,2), size=(self.image_size, self.image_size), mode='nearest'
            # )
            # vc_mask = rearrange(vc_mask, '(b k) c h w -> b k h w c', b=B, k=K)
            # batch['smpl_mask'] = vc_mask.squeeze(-1)
            # batch['vc_smpl_maps'] = vc_maps


            # normalise smpl joints, scan vertices, and smpl vertices
            if self.cfg.DATA.NORMALISE:
                normalise_to_height = 1.7 
                smpl_T_height = smpl_T_output.vertices[:, :, 1].max(dim=-1).values - smpl_T_output.vertices[:, :, 1].min(dim=-1).values
                smpl_T_joints = smpl_T_output.joints[:, :self.num_joints] * (normalise_to_height / smpl_T_height)[:, None, None]
                scan_verts = [verts * (normalise_to_height / smpl_T_height[i]) for i, verts in enumerate(scan_verts)]


            batch['smpl_T_joints'] = rearrange(smpl_T_joints, '(b k) j c -> b k j c', b=B, k=K)
            batch['pose'] = rearrange(smpl_output.full_pose, '(b k) c -> b k c', b=B, k=K)    



            scan_mesh_pytorch3d = Meshes(
                verts=scan_verts,
                faces=scan_faces
            )
            vp = sample_points_from_meshes(scan_mesh_pytorch3d, self.num_samples)
            vp_ptcld = Pointclouds(points=vp)
            batch['vp_ptcld'] = vp_ptcld
            batch['vp'] = vp

        return batch


    

    def configure_optimizers(self):
        # Only include trainable parameters (dynamically updated based on freezing)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        criterion_params = list(self.criterion.parameters())
        params = trainable_params + criterion_params
        optimizer = optim.Adam(params, lr=self.cfg.TRAIN.LR)

        # Optional warmup (linear) followed by cosine annealing
        if self.cfg.TRAIN.LR_SCHEDULER == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.TRAIN.NUM_EPOCHS,
                eta_min=self.cfg.TRAIN.LR * 0.1
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

    def _update_optimizer_parameters(self):
        """Update optimizer parameters when training mode changes"""
        # Get current trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        criterion_params = list(self.criterion.parameters())
        current_params = set(trainable_params + criterion_params)
        
        # Get optimizer parameter groups
        optimizer = self.optimizers()
        if hasattr(optimizer, 'param_groups'):
            optimizer_params = set()
            for group in optimizer.param_groups:
                optimizer_params.update(group['params'])
            
            # If parameters have changed, update optimizer
            if current_params != optimizer_params:
                # Create new parameter groups
                new_param_groups = [
                    {'params': trainable_params, 'lr': self.cfg.TRAIN.LR},
                    {'params': criterion_params, 'lr': self.cfg.TRAIN.LR}
                ]
                optimizer.param_groups = new_param_groups





    @torch.no_grad()
    def forward_and_visualise(self, batch, batch_idx):

        batch = self.process_4ddress(batch, batch_idx, normalise=self.normalise)

        preds = self(batch)

        self.visualiser.visualise(preds, batch, batch_idx=batch_idx) 


    def build_avatar(self, batch):

        batch = self.process_4ddress(batch, batch_idx=0, normalise=self.normalise)

        preds_vc = self.model._forward_vc(batch)

        return preds_vc 

    def drive_avatar(self, vc, batch, novel_pose):
        preds_vp = self.model._forward_vp(vc, batch, novel_pose)
        return preds_vp 




    def knn_ptcld(self, x, y, K=1):
        with torch.autocast(enabled=False, device_type='cuda'):
            x_lengths, x_normals = None, None
            y_lengths, y_normals = None, None

            x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
            y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
            
            x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=2, K=K)
            dist, idx, _ = x_nn
            return dist, idx
   




    def _freeze_canonical_modules(self):
        for param in self.model.aggregator.parameters():
            param.requires_grad = False
        for param in self.model.canonical_head.parameters():
            param.requires_grad = False
        if hasattr(self.model, 'skinning_head'):
            for param in self.model.skinning_head.parameters():
                param.requires_grad = False
        if self.use_sapiens:
            for param in self.model.sapiens.parameters():
                param.requires_grad = False
            
        print("freeze canonical stage")

    def _unfreeze_canonical_modules(self):
        for param in self.model.aggregator.parameters():
            param.requires_grad = True
        for param in self.model.canonical_head.parameters():
            param.requires_grad = True
        if hasattr(self.model, 'skinning_head'):
            for param in self.model.skinning_head.parameters():
                param.requires_grad = True
        for module in [self.smpl_male, self.smpl_female, self.smpl_neutral]:
            for param in module.parameters():
                param.requires_grad = False


    def _freeze_pbs_modules(self):
        for module in [self.model.pbs_aggregator, self.model.pbs_head]:
            for param in module.parameters():
                param.requires_grad = False

    def _unfreeze_pbs_modules(self):
        for param in self.model.pbs_aggregator.parameters():
            param.requires_grad = True
        for param in self.model.pbs_head.parameters():
            param.requires_grad = True
        for module in [self.smpl_male, self.smpl_female, self.smpl_neutral]:
            for param in module.parameters():
                param.requires_grad = False

    def _set_train_vc(self):
        self._unfreeze_canonical_modules()
        self._freeze_pbs_modules()
        print("train canonical stage, freeze pbs stage")

    def _set_train_vp(self):
        self._freeze_canonical_modules()
        self._unfreeze_pbs_modules()
        print("train pbs stage, freeze canonical stage")
    

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

        # Set equal aspect ratio for both plots
        ax1.set_box_aspect([1,1,1])
        ax2.set_box_aspect([1,1,1])

        # Make ticks equal scale
        ax1.set_aspect('equal', adjustable='box')
        ax2.set_aspect('equal', adjustable='box')
        
        

        plt.tight_layout()
        # plt.show()
        plt.savefig('smpl_scan_alignment.png')
        plt.close()
        import ipdb; ipdb.set_trace()

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








    # # Create scatter plot comparing scan and SMPL vertices
    # import matplotlib.pyplot as plt

    # # Create 3D figure
    # fig = plt.figure(figsize=(12, 6))
    
    # # SMPL vertices plot
    # ax1 = fig.add_subplot(121, projection='3d')
    # smpl_verts = smpl_vertices[0,0].detach().cpu().numpy()
    # ax1.scatter(smpl_verts[:,0], smpl_verts[:,1], smpl_verts[:,2], s=0.1, c='blue', marker='.', alpha=0.6)
    # ax1.set_title('SMPL Vertices')
    # ax1.set_xlabel('X')
    # ax1.set_ylabel('Y')
    # ax1.set_zlabel('Z')

    # # Scan vertices plot 
    # ax2 = fig.add_subplot(122, projection='3d')
    # scan_verts = scan_mesh_verts[0].detach().cpu().numpy()
    # ax2.scatter(scan_verts[:,0], scan_verts[:,1], scan_verts[:,2], s=0.1, c='red', marker='.', alpha=0.6)
    # ax2.scatter(smpl_verts[:,0], smpl_verts[:,1], smpl_verts[:,2], s=0.1, c='blue', marker='.', alpha=0.6)
    # ax2.set_title('Scan Vertices')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')

    # # Set equal aspect ratio for both plots
    # ax1.set_box_aspect([1,1,1])
    # ax2.set_box_aspect([1,1,1])

    # # Make ticks equal scale
    # ax1.set_aspect('equal', adjustable='box')
    # ax2.set_aspect('equal', adjustable='box')

    # # Set view to look into z-axis
    # ax1.view_init(elev=10, azim=10, vertical_axis='y')
    # ax2.view_init(elev=10, azim=10, vertical_axis='y')

    # plt.tight_layout()
    # plt.savefig('vertices_comparison.png')
    # plt.close()

    # import ipdb; ipdb.set_trace()









    # # Create 2D plot comparing SMPL and template vertices
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(18, 6))

    # a, b = 0, 1

    # # Plot SMPL vertices
    # plt.subplot(131)
    # plt.scatter(smpl_T_vertices[0, 0, :, a].cpu().numpy(), 
    #            smpl_T_vertices[0, 0, :, b].cpu().numpy(),
    #            c='blue', alpha=0.5, s=1)
    # plt.title('SMPL T-Pose Vertices')
    # plt.axis('equal')

    # # Plot template vertices 
    # plt.subplot(132)
    # plt.scatter(batch['template_mesh_verts'][0][:, a].cpu().numpy(),
    #            batch['template_mesh_verts'][0][:, b].cpu().numpy(), 
    #            c='red', alpha=0.5, s=1)
    # plt.title('Template Mesh Vertices')
    # plt.axis('equal')

    # plt.subplot(133)
    # plt.scatter(smpl_T_vertices[0, 0, :, a].cpu().numpy(), 
    #            smpl_T_vertices[0, 0, :, b].cpu().numpy(),
    #            c='blue', alpha=0.5, s=0.5)
    # plt.scatter(batch['template_mesh_verts'][0][:, a].cpu().numpy(),
    #            batch['template_mesh_verts'][0][:, b].cpu().numpy(), 
    #            c='red', alpha=0.5, s=0.5)
    # plt.axis('equal')

    # # Get min/max ranges for both plots
    # smpl_x_min = smpl_T_vertices[0, 0, :, a].cpu().numpy().min()
    # smpl_x_max = smpl_T_vertices[0, 0, :, a].cpu().numpy().max()
    # smpl_y_min = smpl_T_vertices[0, 0, :, b].cpu().numpy().min() 
    # smpl_y_max = smpl_T_vertices[0, 0, :, b].cpu().numpy().max()

    # template_x_min = batch['template_mesh_verts'][0][:, a].cpu().numpy().min()
    # template_x_max = batch['template_mesh_verts'][0][:, a].cpu().numpy().max()
    # template_y_min = batch['template_mesh_verts'][0][:, b].cpu().numpy().min()
    # template_y_max = batch['template_mesh_verts'][0][:, b].cpu().numpy().max()

    # # Set same range for both plots
    # x_min = min(smpl_x_min, template_x_min)
    # x_max = max(smpl_x_max, template_x_max)
    # y_min = min(smpl_y_min, template_y_min)
    # y_max = max(smpl_y_max, template_y_max)

    # plt.subplot(131).set_xlim(x_min, x_max)
    # plt.subplot(131).set_ylim(y_min, y_max)
    # plt.subplot(132).set_xlim(x_min, x_max)
    # plt.subplot(132).set_ylim(y_min, y_max)
    # plt.subplot(133).set_xlim(x_min, x_max)
    # plt.subplot(133).set_ylim(y_min, y_max)

    # plt.tight_layout()
    # plt.savefig(f'vertex_comparison.png')
    # plt.close()

    # import ipdb; ipdb.set_trace()


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
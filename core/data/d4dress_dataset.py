import os
import torch
import smplx 
import pickle
import random
import numpy as np
from PIL import Image
import trimesh
from typing import Sequence
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from core.configs.paths import BASE_PATH, DATA_PATH as PATH_TO_DATASET
from core.configs.model_size_cfg import MODEL_CONFIGS

import sys
sys.path.append('.')

from core.data.d4dress_utils import load_pickle, load_image, rotation_matrix, d4dress_cameras_to_pytorch3d_cameras



# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class D4DressDataset(Dataset):
    def __init__(self, cfg, ids):
        self.cfg = cfg
        self.num_frames_pp = getattr(cfg.DATA, 'NUM_FRAMES_PP', 4)
        assert self.num_frames_pp >= 2, "NUM_FRAMES_PP must be at least 2"
        self.lengthen_by = cfg.DATA.LENGHTHEN_D4DRESS

        self.img_size = cfg.DATA.IMAGE_SIZE
        self.body_model = cfg.MODEL.BODY_MODEL
        self.num_joints = 24 if self.body_model == 'smpl' else 55

        self.subject_ids = ids 
        self.layer = ['Inner', 'Outer']
        self.camera_ids = ['0004', '0028', '0052', '0076']

        self.takes = defaultdict(list)
        self.num_of_takes = defaultdict(int)
        
        # Pre-load template meshes and weights for all subjects and layers
        self.template_meshes = defaultdict(dict)  # {id: {layer: {mesh_type: mesh}}}
        self.template_lbs_weights = defaultdict(dict)  # {id: {layer: weights}}
        self.smpl_T_data = defaultdict(dict)  # {id: {joints: ..., vertices: ...}}
        
        for subject_id in self.subject_ids:
            template_dir = os.path.join(PATH_TO_DATASET, '_4D-DRESS_Template', subject_id)
            
            # Load SMPL T data (same for both layers)
            smpl_T_joints = np.load(os.path.join(template_dir, 'smpl_T_joints.npy'))
            smpl_T_vertices = np.load(os.path.join(template_dir, 'smpl_T_vertices.npy'))
            self.smpl_T_data[subject_id] = {
                'joints': torch.tensor(smpl_T_joints, dtype=torch.float32)[:, :self.num_joints],
                'vertices': torch.tensor(smpl_T_vertices, dtype=torch.float32)
            }
            
            # Load template meshes for each layer
            for layer in self.layer:
                suffix = '_w_outer' if layer == 'Outer' else ''
                
                # Load filtered mesh
                full_filtered_mesh = trimesh.load(os.path.join(template_dir, f'filtered{suffix}.ply'))
                full_filtered_vertices = torch.tensor(full_filtered_mesh.vertices, dtype=torch.float32)
                full_filtered_faces = torch.tensor(full_filtered_mesh.faces, dtype=torch.long)
                
                # Load full mesh and LBS weights
                template_full_mesh = trimesh.load(os.path.join(template_dir, f'full_mesh{suffix}.ply'))
                template_full_lbs_weights = torch.tensor(
                    np.load(os.path.join(template_dir, f'full_lbs_weights{suffix}.npy')), 
                    dtype=torch.float32
                )
                
                self.template_meshes[subject_id][layer] = {
                    'filtered_mesh': full_filtered_mesh,
                    'filtered_vertices': full_filtered_vertices,
                    'filtered_faces': full_filtered_faces,
                    'full_mesh': template_full_mesh,
                    'full_lbs_weights': template_full_lbs_weights
                }
            
            inner_takes = sorted(os.listdir(os.path.join(PATH_TO_DATASET, subject_id, self.layer[0])))
            inner_takes = [(take, 'Inner') for take in inner_takes if take.startswith('Take')]
            outer_takes = sorted(os.listdir(os.path.join(PATH_TO_DATASET, subject_id, self.layer[1])))
            outer_takes = [(take, 'Outer') for take in outer_takes if take.startswith('Take')]

            self.takes[subject_id] = inner_takes #+ outer_takes
            self.num_of_takes[subject_id] = len(inner_takes )#+ outer_takes)


        self.transform = transforms.Compose([
            transforms.CenterCrop((940, 940)),
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.CenterCrop((940, 940)),
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        self.sapiens_transform = transforms.Compose([
            transforms.CenterCrop((940, 940)),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        

    def __len__(self):
        return int(len(self.subject_ids) * self.lengthen_by)

    def __getitem__(self, index):
        # Create a deterministic random state for this specific index
        # This ensures the same index always produces the same random samples
        rng_state = torch.get_rng_state()
        np_rng_state = np.random.get_state()
        
        # Seed based on index for deterministic sampling
        torch.manual_seed(42 + index)
        np.random.seed(42 + index)
        
        ret = defaultdict(list)
        ret['dataset'] = '4DDress'

        subject_id = self.subject_ids[index // self.lengthen_by]

        num_of_takes = self.num_of_takes[subject_id]
        sampled_take = self.takes[subject_id][torch.randint(0, num_of_takes, (1,)).item()]
        take_dir = os.path.join(PATH_TO_DATASET, subject_id, sampled_take[1], sampled_take[0])

        # Sample an extra take in the same layer as sampled_take
        same_layer_takes = [t for t in self.takes[subject_id] if t[1] == sampled_take[1]]
        if len(same_layer_takes) > 1:
            extra_take = sampled_take
            while extra_take == sampled_take:
                extra_take = same_layer_takes[torch.randint(0, len(same_layer_takes), (1,)).item()]
        else:
            extra_take = sampled_take  # fallback if only one take in this layer

        extra_take_dir = os.path.join(PATH_TO_DATASET, subject_id, extra_take[1], extra_take[0])

        
        suffix = '_w_outer' if sampled_take[1] == 'Outer' else ''

        ret['take_dir'] = take_dir
        ret['scan_ids'] = subject_id   

        # Load basic_info from main take
        basic_info = load_pickle(os.path.join(take_dir, 'basic_info.pkl'))
        gender = basic_info['gender'] # is str
        ret['gender'] = gender
        scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
        
        # Load basic_info from extra take for the last frame
        extra_basic_info = load_pickle(os.path.join(extra_take_dir, 'basic_info.pkl'))
        extra_scan_frames, extra_scan_rotation = extra_basic_info['scan_frames'], extra_basic_info['rotation']
        
        # Sample frames: first num_frames_pp from main take, last one from extra take
        sampled_frames_main = np.random.choice(scan_frames, size=self.num_frames_pp, replace=False)
        sampled_frame_extra = np.random.choice(extra_scan_frames, size=1, replace=False)[0]
        sampled_frames = list(sampled_frames_main) + [sampled_frame_extra]
        
        # Sample cameras to match number of frames (num_frames_pp + 1 total)
        # Use first num_frames_pp cameras from fixed list, then one random for the extra frame
        if self.num_frames_pp <= len(self.camera_ids):
            sampled_cameras = list(self.camera_ids[:self.num_frames_pp]) + [np.random.choice(self.camera_ids, size=1, replace=False).item()]
        else:
            # If num_frames_pp > number of available cameras, repeat cameras
            num_repeats = self.num_frames_pp - len(self.camera_ids)
            sampled_cameras = list(self.camera_ids) + [
                np.random.choice(self.camera_ids, size=1, replace=False).item() 
                for _ in range(num_repeats + 1)
            ]
        
        # Restore the original random state
        torch.set_rng_state(rng_state)
        np.random.set_state(np_rng_state)
        
        # Load camera params from both takes
        camera_params = load_pickle(os.path.join(take_dir, 'Capture', 'cameras.pkl'))
        extra_camera_params = load_pickle(os.path.join(extra_take_dir, 'Capture', 'cameras.pkl'))
        
        # Build R, T, K for all cameras: first num_frames_pp from main take, last from extra take
        R_list, T_list, K_list = [], [], []
        for cam_idx, cam_id in enumerate(sampled_cameras):
            if cam_idx == len(sampled_cameras) - 1:  # Last camera uses extra take
                R_single, T_single, K_single = d4dress_cameras_to_pytorch3d_cameras(extra_camera_params, [cam_id])
            else:  # Other cameras use main take
                R_single, T_single, K_single = d4dress_cameras_to_pytorch3d_cameras(camera_params, [cam_id])
            R_list.append(torch.from_numpy(R_single[0]))
            T_list.append(torch.from_numpy(T_single[0]))
            K_list.append(torch.from_numpy(K_single[0]))

        ret['R'] = torch.stack(R_list)
        ret['T'] = torch.stack(T_list)
        ret['K'] = torch.stack(K_list)


        # ---- template mesh (use pre-loaded data) ----
        layer = sampled_take[1]
        template_data = self.template_meshes[subject_id][layer]


        ret['template_mesh'] = template_data['filtered_mesh']
        ret['template_mesh_verts'] = template_data['filtered_vertices']
        ret['template_mesh_faces'] = template_data['filtered_faces']
        
        ret['template_full_mesh'] = template_data['full_mesh']
        ret['template_full_lbs_weights'] = template_data['full_lbs_weights']
        # for key in ['filtered_mesh', 'filtered_vertices', 'filtered_faces', 'full_mesh', 'full_lbs_weights']:
        #     ret[str('template_') + key] = template_data[key]


        # -------------- SMPL T joints and vertices (use pre-loaded data) --------------
        smpl_T_data = self.smpl_T_data[subject_id]
        ret['smpl_T_joints'] = smpl_T_data['joints']
        ret['smpl_T_vertices'] = smpl_T_data['vertices']


        for i, sampled_frame in enumerate(sampled_frames):
            # Determine which take to use: last frame uses extra_take, others use main take
            is_last_frame = (i == len(sampled_frames) - 1)
            current_take_dir = extra_take_dir if is_last_frame else take_dir
            current_scan_rotation = extra_scan_rotation if is_last_frame else scan_rotation
            
            # ---- smpl / smplx data ----
            smpl_data_fname = os.path.join(current_take_dir, self.body_model.upper(), 'mesh-f{}_{}.pkl'.format(sampled_frame, self.body_model))
            smpl_data = load_pickle(smpl_data_fname)

            for key in ['global_orient', 'body_pose', 'transl', 'betas']:
                ret[key].append(smpl_data[key])

            if self.body_model == 'smplx': 
                for key in ['left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                    ret[key].append(smpl_data[key])
                pose = np.concatenate(
                    [smpl_data['global_orient'], smpl_data['body_pose'], 
                     smpl_data['jaw_pose'], smpl_data['leye_pose'], smpl_data['reye_pose'], 
                     smpl_data['left_hand_pose'], smpl_data['right_hand_pose']], axis=0
                )
            elif self.body_model == 'smpl':
                pose = np.concatenate([smpl_data['global_orient'], smpl_data['body_pose']], axis=0)
            else:
                raise ValueError(f"Body model {self.body_model} not supported")
            ret['pose'].append(pose)

            
            # ---- scan mesh ----
            scan_mesh_fname = os.path.join(current_take_dir, 'Meshes_pkl', 'mesh-f{}.pkl'.format(sampled_frame))
            scan_mesh = load_pickle(scan_mesh_fname)
            scan_mesh['uv_path'] = scan_mesh_fname.replace('mesh-f', 'atlas-f')

            ret['scan_mesh'].append(scan_mesh)
            ret['scan_rotation'].append(torch.tensor(current_scan_rotation).float())
            ret['scan_mesh_verts'].append(torch.tensor(scan_mesh['vertices']).float()) # - transl[None, :]).float())
            ret['scan_mesh_verts_centered'].append(torch.tensor(scan_mesh['vertices'] - smpl_data['transl'][None, :]).float())
            ret['scan_mesh_faces'].append(torch.tensor(scan_mesh['faces']).long())
            ret['scan_mesh_colors'].append(torch.tensor(scan_mesh['colors']).float())

            # ---- images ----
            img_fname = os.path.join(current_take_dir, 'Capture', sampled_cameras[i], 'images', 'capture-f{}.png'.format(sampled_frame))
            mask_fname = os.path.join(current_take_dir, 'Capture', sampled_cameras[i], 'masks', 'mask-f{}.png'.format(sampled_frame))
            
            # Load images and apply transforms
            img = load_image(img_fname)
            mask = load_image(mask_fname)

            masked_img = img * mask[..., None]
            
            # Convert to PIL Images for transforms
            img_pil = Image.fromarray(img)
            mask_pil = Image.fromarray(mask)
            masked_img_pil = Image.fromarray(masked_img)
            
            img_transformed = self.transform(img_pil)
            mask_transformed = self.mask_transform(mask_pil).squeeze()

            sapiens_image = self.sapiens_transform(masked_img_pil)

            ret['sapiens_images'].append(sapiens_image)
            ret['imgs'].append(img_transformed)
            ret['masks'].append(mask_transformed)


        for key in list(ret.keys()):
            if isinstance(ret[key], (list, tuple)) and len(ret[key]) > 0:
                try:
                    ret[key] = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in ret[key]])
                except:
                    pass
        return ret







if __name__ == "__main__":
    from pytorch_lightning import seed_everything
    import sys
    sys.path.append('.')
    from core.configs.cch_cfg import get_cch_cfg_defaults

    seed_everything(42)
    dataset = D4DressDataset(cfg=get_cch_cfg_defaults(), ids=['00138', '00191'])

    # Example of how to use the dataset with custom collate function
    # This prevents PyTorch from trying to stack meshes with different vertex counts
    from torch.utils.data import DataLoader
    

    def custom_collate_fn(batch):
        """
        Custom collate function to handle variable-sized mesh data.
        Returns lists for mesh data that can't be stacked, and tensors for data that can.
        """
        collated = defaultdict(list)
        
        for sample in batch:
            for key, value in sample.items():
                collated[key].append(value)
        
        nonstackable_keys = [
            #THuman
            'scan_verts', 'scan_faces', 'smplx_param', 'gender', 'scan_ids', 'camera_ids', 'dataset',
            #4DDress
            'scan_mesh', 'scan_mesh_verts', 'scan_mesh_faces', 'scan_mesh_verts_centered', 'scan_mesh_colors',
            'template_mesh', 'template_mesh_verts', 'template_mesh_faces', 'template_full_mesh', 
            'template_full_lbs_weights', 'gender', 'take_dir', 'dataset'
        ]


        for key in collated.keys():
            if collated[key] and (key not in nonstackable_keys):
                try:
                    collated[key] = torch.stack(collated[key])
                except RuntimeError as e:
                    print(f"Warning: Could not stack {key}, keeping as list. Error: {e}")
                    # Keep as list if stacking fails
        
        # Keep mesh data as lists since they have different vertex counts
        for key in nonstackable_keys:
            if key in collated:
                # Keep as list - don't try to stack
                pass
        
        return dict(collated)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    for batch in dataloader:
        for k, v in batch.items():
            print(k, v.shape)
        import ipdb; ipdb.set_trace()
    
    # print("Testing dataset with custom collate function...")
    # for batch_idx, batch in enumerate(dataloader):
    #     print(f"\nBatch {batch_idx}:")
    #     for k, v in batch.items():
    #         if isinstance(v, torch.Tensor):
    #             print(f"  {k}: tensor shape {v.shape}")
    #         else:
    #             print(f"  {k}: list of {len(v)} items")
    #             if k in ['scan_mesh', 'smpl_data'] and v:
    #                 # Show some info about the first mesh
    #                 first_mesh = v[0][0] if isinstance(v[0], list) else v[0]
    #                 if isinstance(first_mesh, dict) and 'vertices' in first_mesh:
    #                     print(f"    First mesh has {len(first_mesh['vertices'])} vertices")
        
    #     if batch_idx >= 1:  # Only test first few batches
    #         break


    # print(f"batch keys: {batch.keys()}")
    # print(f"batch['imgs'].shape: {batch['imgs'].shape}")
    # print(f"batch['masks'].shape: {batch['masks'].shape}")
    # print(type(batch['scan_mesh'][0]))
    


    # # Visualize images from the batch
    # import matplotlib.pyplot as plt

    # batch_size, num_views = batch['imgs'].shape[:2]
    # fig, axes = plt.subplots(batch_size, num_views, figsize=(4*num_views, 4*batch_size))
    
    # for b in range(batch_size):
    #     for n in range(num_views):
    #         img = batch['imgs'][b,n].numpy()
    #         if batch_size == 1:
    #             ax = axes[n]
    #         else:
    #             ax = axes[b,n]
    #         ax.imshow(img)
    #         ax.axis('off')
    #         ax.set_title(f'Batch {b}, View {n}')
    
    # plt.tight_layout()
    # plt.show()



    import ipdb; ipdb.set_trace()
    print('')
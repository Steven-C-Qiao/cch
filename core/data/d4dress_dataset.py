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


# PATH_TO_DATASET = os.path.join(BASE_PATH, "4DDress")

"""
4D-DRESS
└── < Subject ID > (00***)
    └── < Outfit > (Inner, Outer)
       └── < Sequence ID > (Take*)
            ├── basic_info.pkl: {'scan_frames', 'rotation', 'offset', ...}
            ├── Meshes_pkl
            │   ├── atlas-fxxxxx.pkl: uv texture map as pickle file (1024, 1024, 3)
            │   └── mesh-fxxxxx.pkl: {'vertices', 'faces', 'colors', 'normals', 'uvs'}
            ├── SMPL
            │   ├── mesh-fxxxxx_smpl.pkl: SMPL params
            │   └── mesh-fxxxxx_smpl.ply: SMPL mesh
            ├── SMPLX
            │   ├── mesh-fxxxxx_smplx.pkl: SMPLX params
            │   └── mesh-fxxxxx_smplx.ply: SMPLX mesh
            ├── Semantic
            │   ├── labels
            │   │   └── label-fxxxxx.pkl, {'scan_labels': (nvt, )}
            │   ├── clothes: let user extract
            │   │   └── cloth-fxxxxx.pkl, {'upper': {'vertices', 'faces', 'colors', 'uvs', 'uv_path'}, ...}
            ├── Capture
            │   ├── cameras.pkl: {'cam_id': {"intrinsics", "extrinsics", ...}}
            │   ├── < Camera ID > (0004, 0028, 0052, 0076)
            │   │   ├── images
            │   │   │   └── capture-f*****.png: captured image (1280, 940, 3) 
            │   │   ├── masks
            │   │   │   └── mask-f*****.png: rendered mask (1280, 940)
            │   │   ├── labels: let user extract
            │   │   │   └── label-f*****.png: rendered label (1280, 940, 3) 
            └── └── └── └── overlap-f*****.png: overlapped label (1280, 940, 3)
"""


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def d4dress_collate_fn(batch):
    """
    Custom collate function to handle variable-sized mesh data.
    Returns lists for mesh data that can't be stacked, and tensors for data that can.
    """
    collated = defaultdict(list)
    
    for sample in batch:
        for key, value in sample.items():
            collated[key].append(value)
    
    nonstackable_keys = [
        'scan_mesh', 'scan_mesh_verts', 'scan_mesh_faces', 'scan_mesh_verts_centered', 'scan_mesh_colors',
        'template_mesh', 'template_mesh_verts', 'template_mesh_faces', 'template_full_mesh', 'template_full_lbs_weights', 'gender', 'take_dir']


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






class D4DressDataset(Dataset):
    def __init__(self, cfg, ids):
        self.cfg = cfg
        self.num_frames_pp = 4
        self.lengthen_by = 500

        self.img_size = cfg.DATA.IMAGE_SIZE
        self.body_model = cfg.MODEL.BODY_MODEL
        self.num_joints = 24 if self.body_model == 'smpl' else 55

        self.ids = ids 
        self.layer = 'Inner'
        self.camera_ids = ['0004', '0028', '0052', '0076']

        self.takes = defaultdict(list)
        self.num_of_takes = defaultdict(int)
        for id in self.ids:
            takes = os.listdir(os.path.join(PATH_TO_DATASET, id, self.layer))
            takes = [take for take in takes if take.startswith('Take')]
            self.takes[id] = takes
            self.num_of_takes[id] = len(takes)


        self.transform = transforms.Compose([
            transforms.CenterCrop((940, 940)),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # make_normalize_transform()
        ])
        self.mask_transform = transforms.Compose([
            transforms.CenterCrop((940, 940)),
            transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            # make_normalize_transform()
        ])
        self.normalise = make_normalize_transform()

        self.sapiens_transform = sapiens_transform

        

    def __len__(self):
        return len(self.ids) * self.lengthen_by

    def __getitem__(self, index):
        ret = defaultdict(list)

        id = self.ids[index // self.lengthen_by]
        layer = self.layer

        num_of_takes = self.num_of_takes[id]
        sampled_take = self.takes[id][torch.randint(0, num_of_takes, (1,)).item()]
        take_dir = os.path.join(PATH_TO_DATASET, id, layer, sampled_take)

        ret['take_dir'] = take_dir

        basic_info = load_pickle(os.path.join(take_dir, 'basic_info.pkl'))
        gender = basic_info['gender'] # is str
        ret['gender'] = gender
        scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
        sampled_frames = np.random.choice(scan_frames, size=self.num_frames_pp + 1, replace=False)
        
        sampled_cameras = self.camera_ids + [np.random.choice(self.camera_ids, size=1, replace=False).item()]
        camera_params = load_pickle(os.path.join(take_dir, 'Capture', 'cameras.pkl'))
        R, T, K = d4dress_cameras_to_pytorch3d_cameras(camera_params, sampled_cameras)
        ret['R'] = R
        ret['T'] = T
        ret['K'] = K


        # ---- template mesh ----
        template_dir = os.path.join(PATH_TO_DATASET, '_4D-DRESS_Template', id)
            
        upper_mesh = trimesh.load(os.path.join(template_dir, 'upper.ply'))
        body_mesh = trimesh.load(os.path.join(template_dir, 'body.ply'))
        if os.path.exists(os.path.join(template_dir, 'lower.ply')):
            lower_mesh = trimesh.load(os.path.join(template_dir, 'lower.ply'))
            clothing_mesh = trimesh.util.concatenate([lower_mesh, upper_mesh])
        else:
            clothing_mesh = upper_mesh
        # full_mesh = trimesh.util.concatenate([body_mesh, clothing_mesh])


        full_filtered_mesh = trimesh.load(os.path.join(template_dir, 'filtered.ply'))
        full_filtered_vertices = full_filtered_mesh.vertices
        full_filtered_faces = full_filtered_mesh.faces

        # template_mesh_lbs_weights = np.load(os.path.join(template_dir, 'filtered_lbs_weights.npy'))
        
        ret['template_mesh'] = full_filtered_mesh
        # ret['template_body_mesh'] = torch.tensor(body_mesh, dtype=torch.float32)
        ret['template_mesh_verts'] = torch.tensor(full_filtered_vertices, dtype=torch.float32)
        ret['template_mesh_faces'] = torch.tensor(full_filtered_faces, dtype=torch.long)
        # ret['template_mesh_lbs_weights'] = torch.tensor(template_mesh_lbs_weights, dtype=torch.float32)

        template_full_mesh = trimesh.load(os.path.join(template_dir, 'full_mesh.ply'))
        template_full_lbs_weights = np.load(os.path.join(template_dir, 'full_lbs_weights.npy'))
        ret['template_full_mesh'] = template_full_mesh
        ret['template_full_lbs_weights'] = torch.tensor(template_full_lbs_weights, dtype=torch.float32)


        # -------------- SMPL T joints and vertices --------------
        smpl_T_joints = np.load(os.path.join(template_dir, 'smpl_T_joints.npy'))
        smpl_T_vertices = np.load(os.path.join(template_dir, 'smpl_T_vertices.npy'))
        ret['smpl_T_joints'] = torch.tensor(smpl_T_joints, dtype=torch.float32)[:, :self.num_joints]
        ret['smpl_T_vertices'] = torch.tensor(smpl_T_vertices, dtype=torch.float32)


        for i, sampled_frame in enumerate(sampled_frames):
            
            # ---- smpl / smplx data ----
            smpl_data_fname = os.path.join(take_dir, self.body_model.upper(), 'mesh-f{}_{}.pkl'.format(sampled_frame, self.body_model))
            # smpl_ply_fname = os.path.join(take_dir, self.body_model.upper(), 'mesh-f{}_{}.ply'.format(sampled_frame, self.body_model))

            smpl_data = load_pickle(smpl_data_fname)
            global_orient, body_pose, transl, betas = smpl_data['global_orient'], smpl_data['body_pose'], smpl_data['transl'], smpl_data['betas']
            ret['global_orient'].append(global_orient)
            ret['body_pose'].append(body_pose)
            ret['transl'].append(transl)
            ret['betas'].append(betas)

            if self.body_model == 'smplx': # additionally load smplx attributes
                ret['left_hand_pose'].append(smpl_data['left_hand_pose'])
                ret['right_hand_pose'].append(smpl_data['right_hand_pose'])
                ret['jaw_pose'].append(smpl_data['jaw_pose'])
                ret['leye_pose'].append(smpl_data['leye_pose'])
                ret['reye_pose'].append(smpl_data['reye_pose'])
                ret['expression'].append(smpl_data['expression'])
                pose = np.concatenate(
                    [global_orient, body_pose, 
                     smpl_data['jaw_pose'], smpl_data['leye_pose'], smpl_data['reye_pose'], 
                     smpl_data['left_hand_pose'], smpl_data['right_hand_pose']], axis=0
                )

               

            elif self.body_model == 'smpl':
                pose = np.concatenate([global_orient, body_pose], axis=0)
            else:
                raise ValueError(f"Body model {self.body_model} not supported")

            ret['pose'].append(pose)

            
            # ---- scan mesh ----
            scan_mesh_fname = os.path.join(take_dir, 'Meshes_pkl', 'mesh-f{}.pkl'.format(sampled_frame))
            scan_mesh = load_pickle(scan_mesh_fname)
            scan_mesh['uv_path'] = scan_mesh_fname.replace('mesh-f', 'atlas-f')

            ret['scan_mesh'].append(scan_mesh)
            ret['scan_rotation'].append(torch.tensor(scan_rotation).float())
            ret['scan_mesh_verts'].append(torch.tensor(scan_mesh['vertices']).float()) # - transl[None, :]).float())
            ret['scan_mesh_verts_centered'].append(torch.tensor(scan_mesh['vertices'] - transl[None, :]).float())
            ret['scan_mesh_faces'].append(torch.tensor(scan_mesh['faces']).long())
            ret['scan_mesh_colors'].append(torch.tensor(scan_mesh['colors']).float())

            # ---- images ----
            img_fname = os.path.join(take_dir, 'Capture', sampled_cameras[i], 'images', 'capture-f{}.png'.format(sampled_frame))
            mask_fname = os.path.join(take_dir, 'Capture', sampled_cameras[i], 'masks', 'mask-f{}.png'.format(sampled_frame))
            
            # Load images and apply transforms
            img = load_image(img_fname)
            mask = load_image(mask_fname)

            masked_img = img * mask[..., None]
            
            # Convert to PIL Images for transforms
            img_pil = Image.fromarray(img)
            mask_pil = Image.fromarray(mask)
            masked_img_pil = Image.fromarray(masked_img)
            
            # Apply transforms
            # img_transformed = self.normalise(self.transform(img_pil))
            img_transformed = self.transform(img_pil)
            mask_transformed = self.mask_transform(mask_pil).squeeze()

            sapiens_image = sapiens_transform(masked_img_pil)

            ret['sapiens_images'].append(sapiens_image)
            ret['imgs'].append(img_transformed)
            ret['masks'].append(mask_transformed)


            # lbs_weights_fname = os.path.join(take_dir, 'Capture', sampled_cameras[i], 'lbs_images', 'lbs_image-f{}.pt'.format(sampled_frame))
            # lbs_weights_fname = '/scratch/u5au/chexuan.u5au/4DDress/00122/Inner/Take2/Capture/0076/lbs_images/lbs_image-f00011.pt'
            # lbs_weights = torch.load(lbs_weights_fname, map_location='cpu')
            # print(f"Loaded lbs_weights type: {type(lbs_weights)}, is_sparse: {lbs_weights.is_sparse if hasattr(lbs_weights, 'is_sparse') else 'N/A'}")
            # ret['smpl_w_maps'].append(lbs_weights)


        ret['imgs'] = torch.tensor(np.stack(ret['imgs']))
        ret['masks'] = torch.tensor(np.stack(ret['masks']))
        ret['R'] = torch.tensor(np.stack(ret['R']))
        ret['T'] = torch.tensor(np.stack(ret['T']))
        ret['K'] = torch.tensor(np.stack(ret['K']))
        ret['global_orient'] = torch.tensor(np.stack(ret['global_orient']))
        ret['body_pose'] = torch.tensor(np.stack(ret['body_pose']))
        ret['pose'] = torch.tensor(np.stack(ret['pose']))
        ret['transl'] = torch.tensor(np.stack(ret['transl']))
        ret['betas'] = torch.tensor(np.stack(ret['betas']))
        ret['scan_rotation'] = torch.tensor(np.stack(ret['scan_rotation']))
        ret['sapiens_images'] = torch.tensor(np.stack(ret['sapiens_images']))
        # ret['smpl_w_maps'] = torch.stack(ret['smpl_w_maps'])
        if self.body_model == 'smplx':
            ret['left_hand_pose'] = torch.tensor(np.stack(ret['left_hand_pose']))
            ret['right_hand_pose'] = torch.tensor(np.stack(ret['right_hand_pose']))
            ret['jaw_pose'] = torch.tensor(np.stack(ret['jaw_pose']))
            ret['leye_pose'] = torch.tensor(np.stack(ret['leye_pose']))
            ret['reye_pose'] = torch.tensor(np.stack(ret['reye_pose']))
            ret['expression'] = torch.tensor(np.stack(ret['expression']))

        return ret

def sapiens_transform(
    image: torch.tensor
) -> torch.tensor:
    transform = transforms.Compose([
        transforms.CenterCrop((940, 940)),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        make_normalize_transform()
    ])
    image = transform(image)

    return image






if __name__ == "__main__":
    dataset = D4DressDataset()

    # Example of how to use the dataset with custom collate function
    # This prevents PyTorch from trying to stack meshes with different vertex counts
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=d4dress_collate_fn)
    
    print("Testing dataset with custom collate function...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: tensor shape {v.shape}")
            else:
                print(f"  {k}: list of {len(v)} items")
                if k in ['scan_mesh', 'smpl_data'] and v:
                    # Show some info about the first mesh
                    first_mesh = v[0][0] if isinstance(v[0], list) else v[0]
                    if isinstance(first_mesh, dict) and 'vertices' in first_mesh:
                        print(f"    First mesh has {len(first_mesh['vertices'])} vertices")
        
        if batch_idx >= 1:  # Only test first few batches
            break


    print(f"batch keys: {batch.keys()}")
    print(f"batch['imgs'].shape: {batch['imgs'].shape}")
    print(f"batch['masks'].shape: {batch['masks'].shape}")
    print(type(batch['scan_mesh'][0]))
    


    # Visualize images from the batch
    import matplotlib.pyplot as plt

    batch_size, num_views = batch['imgs'].shape[:2]
    fig, axes = plt.subplots(batch_size, num_views, figsize=(4*num_views, 4*batch_size))
    
    for b in range(batch_size):
        for n in range(num_views):
            img = batch['imgs'][b,n].numpy()
            if batch_size == 1:
                ax = axes[n]
            else:
                ax = axes[b,n]
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Batch {b}, View {n}')
    
    plt.tight_layout()
    plt.show()



    import ipdb; ipdb.set_trace()
    print('')
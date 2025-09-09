import os
import torch
import smplx 
import pickle
import random
import kornia 
import numpy as np
from PIL import Image
import trimesh
from typing import Sequence
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

import sys
sys.path.append('.')

from core.data.d4dress_utils import load_pickle, load_image, rotation_matrix, d4dress_cameras_to_pytorch3d_cameras


PATH_TO_DATASET = "/scratches/kyuban/cq244/datasets/4DDress"

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
        'template_mesh', 'template_mesh_verts', 'template_mesh_faces', 'gender']


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



class D4DressDataset(Dataset):
    def __init__(self, debug=False):
        self.debug = debug
        self.num_frames_pp = 4
        self.lengthen_by = 500

        self.ids = ['00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137', 
                    '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156', 
                    '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175', 
                    '00176', '00179', '00180', '00185', '00187', '00188', '00190', '00191']
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
            transforms.Resize((224, 224)),
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

        basic_info = load_pickle(os.path.join(take_dir, 'basic_info.pkl'))
        gender = basic_info['gender'] # is str
        ret['gender'] = gender
        scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
        sampled_frames = np.random.choice(scan_frames, size=self.num_frames_pp, replace=False)
        
        
        sampled_cameras = self.camera_ids # np.random.permutation(self.cameras)
        camera_params = load_pickle(os.path.join(take_dir, 'Capture', 'cameras.pkl'))
        R, T, K = d4dress_cameras_to_pytorch3d_cameras(camera_params)
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
        full_mesh = trimesh.util.concatenate([body_mesh, clothing_mesh])

        visibility_path = os.path.join('/scratches/kyuban/cq244/CCH/cch/model_files/4DDress_visible_vertices', f'{id}.npy')
        visible_vertices = np.load(visibility_path)
        naked_vertices = body_mesh.vertices 
        clothing_vertices = clothing_mesh.vertices
        
        # Create a set of visible vertex indices for efficient lookup
        visible_vertex_set = set(visible_vertices)
        
        # Filter body mesh faces to only include faces where all vertices are visible
        body_faces = body_mesh.faces
        # Use numpy operations to check if all vertices in each face are visible
        face_visibility_mask = np.all(np.isin(body_faces, visible_vertices), axis=1)
        visible_body_faces = body_faces[face_visibility_mask]
        
        # Create mapping from original body vertex indices to new filtered indices
        body_vertex_mapping = np.zeros(len(naked_vertices), dtype=int)
        body_vertex_mapping[visible_vertices] = np.arange(len(visible_vertices))
        
        # Remap face indices for body faces to new vertex indices
        remapped_body_faces = body_vertex_mapping[visible_body_faces]
        
        # Get clothing faces and remap their indices to account for the new vertex ordering
        clothing_faces = clothing_mesh.faces
        num_filtered_body_vertices = len(visible_vertices)
        remapped_clothing_faces = clothing_faces + num_filtered_body_vertices
        
        # Combine filtered body vertices and clothing vertices
        filtered_naked_vertices = naked_vertices[visible_vertices]
        full_filtered_vertices = np.concatenate([filtered_naked_vertices, clothing_vertices])
        
        # Combine remapped faces
        full_filtered_faces = np.concatenate([remapped_body_faces, remapped_clothing_faces])

        full_filtered_mesh = trimesh.Trimesh(
            vertices=full_filtered_vertices,
            faces=full_filtered_faces,
        )
        
        ret['template_mesh'] = full_filtered_mesh
        ret['template_mesh_verts'] = torch.tensor(full_filtered_vertices).float()
        ret['template_mesh_faces'] = torch.tensor(full_filtered_faces).long()
        

        for i, sampled_frame in enumerate(sampled_frames):
            
            # ---- smpl data ----
            smpl_data_fname = os.path.join(take_dir, 'SMPL', 'mesh-f{}_smpl.pkl'.format(sampled_frame))
            # smpl_mesh_fname = os.path.join(take_dir, 'SMPL', 'mesh-f{}_smpl.ply'.format(sampled_frame))

            smpl_data = load_pickle(smpl_data_fname)
            global_orient, body_pose, transl, betas = smpl_data['global_orient'], smpl_data['body_pose'], smpl_data['transl'], smpl_data['betas']
            pose = np.concatenate([global_orient, body_pose], axis=0)
            ret['global_orient'].append(global_orient)
            ret['body_pose'].append(body_pose)
            ret['pose'].append(pose)
            ret['transl'].append(transl)
            ret['betas'].append(betas)
            
            
            # ---- scan mesh ----
            scan_mesh_fname = os.path.join(take_dir, 'Meshes_pkl', 'mesh-f{}.pkl'.format(sampled_frame))
            scan_mesh = load_pickle(scan_mesh_fname)
            scan_mesh['uv_path'] = scan_mesh_fname.replace('mesh-f', 'atlas-f')
            if 'colors' not in scan_mesh:
                print(f"No colors in scan_mesh: {scan_mesh_fname}")
                assert False

                # load atlas data
                atlas_data = load_pickle(scan_mesh['uv_path'])
                # load scan uv_coordinate and uv_image as TextureVisuals
                uv_image = Image.fromarray(atlas_data).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM).convert("RGB")
                texture_visual = trimesh.visual.texture.TextureVisuals(uv=scan_mesh['uvs'], image=uv_image)
                # pack scan data as trimesh
                scan_trimesh = trimesh.Trimesh(
                    vertices=scan_mesh['vertices'],
                    faces=scan_mesh['faces'],
                    vertex_normals=scan_mesh['normals'],
                    visual=texture_visual,
                    process=False,
                )
                scan_mesh['colors'] = scan_trimesh.visual.to_color().vertex_colors
            # rotate scan_mesh to view front
            # if scan_rotation is not None: scan_mesh['vertices'] = np.matmul(scan_rotation, scan_mesh['vertices'].T).T
            
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
            
            # Convert to PIL Images for transforms
            img_pil = Image.fromarray(img)
            mask_pil = Image.fromarray(mask)
            
            # Apply transforms
            img_transformed = self.normalise(self.transform(img_pil))
            mask_transformed = self.transform(mask_pil).squeeze()

            sapiens_image = sapiens_transform(img_pil)

            ret['sapiens_images'].append(sapiens_image)
            ret['imgs'].append(img_transformed)
            ret['masks'].append(mask_transformed)

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

        return ret






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
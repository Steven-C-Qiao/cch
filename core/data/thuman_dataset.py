import os.path as osp
import numpy as np
from PIL import Image
import random
import os, cv2
import trimesh
import torch
# import vedo

from collections import defaultdict

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from core.data.thuman_metadata import THuman_metadata
from core.data.d4dress_utils import load_pickle

PATH_TO_THUMAN = '/scratches/kyuban/cq244/datasets/THuman'





def thuman_collate_fn(batch):
    """
    Custom collate function to handle variable-sized mesh data.
    Returns lists for mesh data that can't be stacked, and tensors for data that can.
    """
    collated = defaultdict(list)
    
    for sample in batch:
        for key, value in sample.items():
            collated[key].append(value)
    
    nonstackable_keys = [
        'scan_verts', 'scan_faces', 'smplx_param', 'gender', 'scan_ids', 'camera_ids'
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



class THumanDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.metadata = THuman_metadata

        self.ids = list(self.metadata.keys())

        self.camera_ids = [
            '000', '010', '020', '030', '040', '050', '060', '070', '080', 
            '090', '100', '110', '120', '130', '140', '150', '160', '170', 
            '180', '190', '200', '210', '220', '230', '240', '250', '260', 
            '270', '280', '290', '300', '310', '320', '330', '340', '350', 
        ]

        self.num_frames_pp = 4
        self.lengthen_by = 500

        self.img_size = cfg.DATA.IMAGE_SIZE
        self.body_model = cfg.MODEL.BODY_MODEL
        self.num_joints = 24 if self.body_model == 'smpl' else 55
        
        # PIL to tensor
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return len(self.ids) 
    
    def __getitem__(self, index):
        ret = defaultdict(list)

        N, K = 4, 5 

        id = self.ids[index]
        gender = self.metadata[id]['gender']
        ret['gender'] = gender

        scans_ids = self.metadata[id]['scans']
        sampled_scan_ids = np.random.choice(scans_ids, size=K, replace=True)
        ret['scan_ids'] = sampled_scan_ids
        
        # Sample one camera ID from each row of camera angles
        sampled_cameras = [
            np.random.choice(['000', '010', '020', '030', '040', '050', '060', '070', '080'], size=1)[0],
            np.random.choice(['090', '100', '110', '120', '130', '140', '150', '160', '170'], size=1)[0], 
            np.random.choice(['180', '190', '200', '210', '220', '230', '240', '250', '260'], size=1)[0],
            np.random.choice(['270', '280', '290', '300', '310', '320', '330', '340', '350'], size=1)[0]
        ]
        if len(sampled_cameras) < K:
            additional_cameras = np.random.choice(self.camera_ids, size=K-len(sampled_cameras), replace=False)
            sampled_cameras.extend(additional_cameras)
        ret['camera_ids'] = sampled_cameras

        for i, scan_id in enumerate(sampled_scan_ids):

            img_fname = os.path.join(PATH_TO_THUMAN, 'render/thuman2_36views', scan_id, 'render', f'{sampled_cameras[i]}.png')
            rgba = Image.open(img_fname).convert('RGBA')
            
            # Extract mask from alpha channel
            mask = rgba.split()[-1]
            img = rgba.convert('RGB')

            img_transformed = self.transform(img)
            mask_transformed = self.transform(mask).squeeze()
            
            # img_transformed = img_transformed * mask_transformed
            
            ret['imgs'].append(img_transformed)
            ret['masks'].append(mask_transformed)

            smplx_fname = os.path.join(PATH_TO_THUMAN, 'smplx', scan_id, f'smplx_param.pkl')
            smplx_param = np.load(smplx_fname, allow_pickle=True)
            smplx_param['left_hand_pose'] = smplx_param['left_hand_pose'][:, :12]
            smplx_param['right_hand_pose'] = smplx_param['right_hand_pose'][:, :12]
            ret['smplx_param'].append(smplx_param)
            for k, v in smplx_param.items():
                ret[k].append(v)

            # this is slow, load pickle instead
            # scan_fname = os.path.join(PATH_TO_THUMAN, 'model', scan_id, f'{scan_id}.obj')
            # scan = trimesh.load(scan_fname, process=False, maintain_order=True, skip_materials=True)
            # ret['scan_mesh'].append(scan) \
            scan_fname = os.path.join(PATH_TO_THUMAN, 'pickled', f'{scan_id}.pkl')
            scan = load_pickle(scan_fname)
            verts = scan['vertices'] / smplx_param['scale'] - smplx_param['transl']
            ret['scan_verts'].append(torch.tensor(verts).float())
            ret['scan_faces'].append(torch.tensor(scan['faces']).long())


        # Attempt to stack each value in ret, keep as is if not stackable
        for key in list(ret.keys()):
            if isinstance(ret[key], (list, tuple)) and len(ret[key]) > 0:
                try:
                    ret[key] = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in ret[key]])
                except:
                        pass

        return ret
    



        def opengl_to_pytorch3d_camera(self, R, T, K):
            """
            Convert OpenGL camera parameters to PyTorch3D camera format.
            
            Args:
                R (torch.Tensor): Rotation matrix in OpenGL format (3x3)
                T (torch.Tensor): Translation vector in OpenGL format (3,)
                K (torch.Tensor): Camera intrinsics matrix (4x4)
                
            Returns:
                R_pytorch3d (torch.Tensor): Rotation matrix in PyTorch3D format
                T_pytorch3d (torch.Tensor): Translation vector in PyTorch3D format 
                K_pytorch3d (torch.Tensor): Camera intrinsics in PyTorch3D format
            """
            # OpenGL to PyTorch3D coordinate system conversion
            # PyTorch3D: +X right, +Y up, +Z front (facing out from camera)
            # OpenGL: +X right, +Y up, -Z front (facing into camera)
            coord_transform = torch.tensor([
                [1, 0, 0],
                [0, 1, 0], 
                [0, 0, -1]
            ], device=R.device)

            # Convert rotation
            R_pytorch3d = R @ coord_transform

            # Convert translation 
            T_pytorch3d = T.clone()
            T_pytorch3d[..., 2] *= -1

            # Camera intrinsics remain the same
            
            S = torch.tensor([  [1, 0,       0],
                                [0, -1,  H - 1],
                                [0,  0,       1]], device=K.device, dtype=K.dtype)
            K_pytorch3d = S @ K[:3, :3] @ S.T

            return R_pytorch3d, T_pytorch3d, K_pytorch3d
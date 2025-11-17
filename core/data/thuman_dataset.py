import os
import torch
import numpy as np
import torchvision.transforms as transforms
from scipy.sparse import load_npz

from PIL import Image
from collections import defaultdict
from typing import Sequence
from torch.utils.data import Dataset

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_axis_angle

from core.configs.paths import THUMAN_PATH
from core.data.thuman_metadata import THuman_metadata
from core.data.d4dress_utils import load_pickle
from core.utils.camera_utils import uv_to_pixel_space, custom_opengl2pytorch3d


def load_w_maps_sparse(w_maps_fname):
    """
    Load sparse w_maps from .npz file and convert back to dense format.
    
    Args:
        w_maps_fname: Path to the .npz file containing sparse matrix
        
    Returns:
        numpy array of shape (512, 512, 55) - dense w_map for single camera
    """
    # Load sparse matrix from .npz file
    sparse_matrix = load_npz(w_maps_fname)
    
    # Convert sparse matrix back to dense format
    dense_data = sparse_matrix.toarray()  # Shape: (512*512, 55)
    
    # Reshape back to original format: (512, 512, 55)
    w_map = dense_data.reshape(512, 512, 55)
    
    return w_map



# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# def make_normalize_transform(
#     mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
#     std: Sequence[float] = IMAGENET_DEFAULT_STD,
# ) -> transforms.Normalize:
#     return transforms.Normalize(mean=mean, std=std)


# def sapiens_transform(
#     image: torch.tensor
# ) -> torch.tensor:
#     transform = transforms.Compose([
#         transforms.Resize((1024, 1024)),
#         transforms.ToTensor(),
#         make_normalize_transform()
#     ])
#     image = transform(image)

#     return image


class THumanDataset(Dataset):
    def __init__(self, cfg, ids=None):
        self.cfg = cfg
        self.lengthen_by = cfg.DATA.LENGHTHEN_THUMAN
        self.metadata = THuman_metadata

        if ids is None:
            self.ids = list(self.metadata.keys())
        else:
            self.ids = ids 

        self.camera_ids = [
            '000', '010', '020', '030', '040', '050', '060', '070', '080', 
            '090', '100', '110', '120', '130', '140', '150', '160', '170', 
            '180', '190', '200', '210', '220', '230', '240', '250', '260', 
            '270', '280', '290', '300', '310', '320', '330', '340', '350', 
        ]

        self.num_frames_pp = 4

        self.img_size = cfg.DATA.IMAGE_SIZE
        self.body_model = cfg.MODEL.BODY_MODEL
        self.num_joints = 24 if self.body_model == 'smpl' else 55
        

        self.crop_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                # no normalise since done in cch_aggregator
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
        self.sapiens_transform  = transforms.Compose(
            [
                transforms.Resize(1024),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

        self.vc_map_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.img_size),
            ]
        )
        self.vc_mask_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.img_size, interpolation=Image.NEAREST),
            ]
        )

    def __len__(self):
        return 0# int(len(self.ids) * self.lengthen_by )
    
    def __getitem__(self, index):
        ret = defaultdict(list)
        ret['dataset'] = 'THuman'

        N, K = 4, 5 

        subject_id = self.ids[index // self.lengthen_by]
        gender = self.metadata[subject_id]['gender']
        ret['gender'] = gender

        scans_ids = self.metadata[subject_id]['scans']
        sampled_scan_ids = np.random.choice(scans_ids, size=K, replace=True)
        ret['scan_ids'] = sampled_scan_ids
        
        # Sample one camera ID from each row of camera angles
        sampled_cameras = [
            np.random.choice(['000', '010', '020'], size=1)[0],
            np.random.choice(['090', '100', '110'], size=1)[0], 
            np.random.choice(['180', '190', '200'], size=1)[0],
            np.random.choice(['270', '280', '290'], size=1)[0]
        ]
        if len(sampled_cameras) < K:
            additional_cameras = np.random.choice(self.camera_ids, size=K-len(sampled_cameras), replace=False)
            sampled_cameras.extend(additional_cameras)
        ret['camera_ids'] = sampled_cameras

        

        for i, scan_id in enumerate(sampled_scan_ids):

            img_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, 'render', f'{sampled_cameras[i]}.png')
            rgba = Image.open(img_fname).convert('RGBA')
            
            # Previous implementation:
            # mask = np.array(rgba.split()[-1])
            # New: Extract mask from alpha channel and binarize (>127)
            alpha = np.array(rgba.split()[-1])
            mask = (alpha > 127).astype(np.uint8) * 255

            # ret['raw_masks'].append(mask)
            img = np.array(rgba.convert('RGB'))

            img_pil = Image.fromarray(img)
            mask_pil = Image.fromarray(mask)

            sapiens_img_transformed = self.sapiens_transform(img_pil)
            img_transformed = self.crop_transform(img_pil)
            mask_transformed = self.mask_transform(mask_pil).squeeze(-3)

            
            ret['imgs'].append(img_transformed)
            ret['masks'].append(mask_transformed)
            ret['sapiens_images'].append(sapiens_img_transformed)


            smplx_fname = os.path.join(THUMAN_PATH, 'smplx', scan_id, f'smplx_param.pkl')
            smplx_param = np.load(smplx_fname, allow_pickle=True)
            smplx_param['left_hand_pose'] = smplx_param['left_hand_pose'][:, :12]
            smplx_param['right_hand_pose'] = smplx_param['right_hand_pose'][:, :12]
            if scan_id >= '0526':
                global_orient = torch.tensor(smplx_param['global_orient'])
                global_orient = matrix_to_euler_angles(axis_angle_to_matrix(global_orient), 'XYZ') + torch.tensor([-torch.pi/2, 0., 0.])
                smplx_param['global_orient'] = matrix_to_axis_angle(euler_angles_to_matrix(global_orient, 'XYZ')).numpy()

            ret['smplx_param'].append(smplx_param)
            for k, v in smplx_param.items():
                ret[k].append(v)

            scan_fname = os.path.join(THUMAN_PATH, 'cleaned', f'{scan_id}.pkl')
            scan = load_pickle(scan_fname)
            # verts = scan['vertices'] / smplx_param['scale'] - smplx_param['transl']
            ret['scan_verts'].append(torch.tensor(scan['scan_verts']).float())
            ret['scan_faces'].append(torch.tensor(scan['scan_faces']).long())


            calib_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, 'calib', f'{sampled_cameras[i]}.txt')
            calib = np.loadtxt(calib_fname)
            extrinsic = calib[:4].reshape(4,4)
            intrinsic = calib[4:].reshape(4,4)

            intrinsic = uv_to_pixel_space(intrinsic)
            intrinsic[0, 2] = 256
            intrinsic[1, 2] = 256

            extrinsic = torch.tensor(extrinsic).float()
            intrinsic = torch.tensor(intrinsic).float()

            cam_R, cam_T = custom_opengl2pytorch3d(extrinsic)
            ret['cam_R'].append(cam_R)
            ret['cam_T'].append(cam_T)

            ret['cam_K'].append(intrinsic)


            # vc_maps_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id,  f'{scan_id}_vc_maps_normalised_170cm.npy')
            # vc_map = np.load(vc_maps_fname)  # (36, 512, 512, 3)

            # # Convert camera ID to index (e.g. '000' -> 0, '010' -> 1, etc)
            # camera_idx = int(sampled_cameras[i]) // 10
            # # Index into vc_map using camera index
            # vc_map = vc_map[camera_idx]  # (512, 512, 3)
            # # Create binary mask where none of the elements are 1 across axis=-1

            # vc_map = self.vc_map_transform(vc_map)
            # assert vc_map.shape == (3, self.img_size, self.img_size)
            # # vc_mask = self.vc_mask_transform(vc_mask).squeeze()
            # vc_mask = (vc_map != 1).all(dim=0)

            # w_maps_dir_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, f'{scan_id}_w_maps_sparse', f'{sampled_cameras[i]}.pt') #.npz')
            # w_map = torch.load(w_maps_dir_fname, map_location='cpu')
            # w_map = w_map.to_dense() 
            w_maps_dir_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, f'{scan_id}_w_maps_sparse', f'{sampled_cameras[i]}.npz')
            w_map = load_w_maps_sparse(w_maps_dir_fname)  # Shape: (512, 512, 55)
            assert w_map.shape == (512, 512, 55)
            w_map = self.vc_map_transform(w_map)
            # ret['vc_smpl_maps'].append(vc_map)
            # ret['smpl_mask'].append(vc_mask)
            ret['smpl_w_maps'].append(w_map)


        # Attempt to stack each value in ret, keep as is if not stackable
        for key in list(ret.keys()):
            if isinstance(ret[key], (list, tuple)) and len(ret[key]) > 0:
                try:
                    ret[key] = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in ret[key]])
                except:
                    pass

        # del vc_map, vc_mask, w_map

        return ret
    
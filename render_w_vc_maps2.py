

import torch
import pytorch_lightning as pl
from collections import defaultdict
from torch.utils.data import DataLoader, DistributedSampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader

import os
import torch
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_axis_angle

from core.configs.paths import THUMAN_PATH
from core.data.thuman_metadata import THuman_metadata
from core.data.d4dress_utils import load_pickle
from core.utils.camera_utils import uv_to_pixel_space, custom_opengl2pytorch3d



class THumanDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.lengthen_by = 20
        self.metadata = THuman_metadata

        self.ids = list(self.metadata.keys())

        self.camera_ids = [
            '000', '010', '020', '030', '040', '050', '060', '070', '080', 
            '090', '100', '110', '120', '130', '140', '150', '160', '170', 
            '180', '190', '200', '210', '220', '230', '240', '250', '260', 
            '270', '280', '290', '300', '310', '320', '330', '340', '350', 
        ]

        self.num_frames_pp = 4
        self.lengthen_by = 20

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
        return len(self.ids) * self.lengthen_by
    
    def __getitem__(self, index):
        ret = defaultdict(list)
        ret['dataset'] = 'THuman'

        N, K = 4, 5 

        id = self.ids[index // self.lengthen_by]
        gender = self.metadata[id]['gender']
        ret['gender'] = gender

        scans_ids = self.metadata[id]['scans']
        sampled_scan_ids = np.random.choice(scans_ids, size=K, replace=True)
        ret['scan_ids'] = sampled_scan_ids
        
        # Sample one camera ID from each row of camera angles
        sampled_cameras = self.camera_ids 
        
        ret['camera_ids'] = sampled_cameras

        for i, scan_id in enumerate(sampled_scan_ids):

            img_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, 'render', f'{sampled_cameras[i]}.png')
            rgba = Image.open(img_fname).convert('RGBA')
            
            # Extract mask from alpha channel
            mask = rgba.split()[-1]
            img = rgba.convert('RGB')

            img_transformed = self.transform(img)
            mask_transformed = self.transform(mask).squeeze()
            
            # img_transformed = img_transformed * mask_transformed
            
            ret['imgs'].append(img_transformed)
            ret['masks'].append(mask_transformed)

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

        # Attempt to stack each value in ret, keep as is if not stackable
        for key in list(ret.keys()):
            if isinstance(ret[key], (list, tuple)) and len(ret[key]) > 0:
                try:
                    ret[key] = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in ret[key]])
                except:
                        pass

        return ret
    

def custom_collate_fn(batch, device=None):
    """
    Custom collate function to handle variable-sized mesh data.
    Returns lists for mesh data that can't be stacked, and tensors for data that can.
    If device is provided, moves tensors to the specified device.
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
                # Move to device if specified
                if device is not None and torch.is_tensor(collated[key]):
                    collated[key] = collated[key].to(device)
            except RuntimeError as e:
                print(f"Warning: Could not stack {key}, keeping as list. Error: {e}")
                # Keep as list if stacking fails
    
    # Keep mesh data as lists since they have different vertex counts
    # But still move individual tensors to device if specified
    for key in nonstackable_keys:
        if key in collated and device is not None:
            # Move individual tensors in the list to device
            for i, item in enumerate(collated[key]):
                if torch.is_tensor(item):
                    collated[key][i] = item.to(device)
    
    return dict(collated)



class AdHocDataModule(pl.LightningDataModule):
    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.train_thuman = None
        self.train_d4dress = None
        self.val_d4dress = None

    def setup(self, stage=None):
        # THuman used for training
        self.train_thuman = THumanDataset(self.cfg)

    def val_dataloader(self):
        # Create a device-aware collate function
        def device_aware_collate_fn(batch):
            return custom_collate_fn(batch, device=self.device)
        
        return DataLoader(
            self.train_thuman,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=device_aware_collate_fn,
        )





from core.models.trainer_4ddress import CCHTrainer
from core.configs.cch_cfg import get_cch_cfg_defaults


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = get_cch_cfg_defaults()

    model = CCHTrainer(
        cfg=cfg,
    ).to(device)
    

    # Pass device to the data module
    datsampler = AdHocDataModule(cfg, device=device)
    datsampler.setup()
    thuman_loader = datsampler.val_dataloader()  # Use val_dataloader method
    
    for batch in thuman_loader:
        # Tensors are already moved to device by the device-aware collate function
        batch = model.process_thuman(batch)
        
        import ipdb; ipdb.set_trace()     







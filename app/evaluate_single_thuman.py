import os
import torch
import pickle
import argparse
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence 
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from loguru import logger
from pytorch_lightning import seed_everything
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.sparse import load_npz


import sys
sys.path.append('.')

from core.models.trainer_4ddress import CCHTrainer
from core.configs.paths import BASE_PATH, DATA_PATH as PATH_TO_DATASET
from core.configs.model_size_cfg import MODEL_CONFIGS
from core.data.d4dress_utils import load_pickle, load_image, rotation_matrix, d4dress_cameras_to_pytorch3d_cameras
from core.configs.cch_cfg import get_cch_cfg_defaults
from core.data.full_dataset import custom_collate_fn
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



IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def _move_to_device(sample, device):
    """Recursively move tensors within nested containers to device."""
    if isinstance(sample, torch.Tensor):
        return sample.to(device)
    if isinstance(sample, dict):
        return {k: _move_to_device(v, device) for k, v in sample.items()}
    if isinstance(sample, list):
        return [_move_to_device(v, device) for v in sample]
    if isinstance(sample, tuple):
        return tuple(_move_to_device(v, device) for v in sample)
    return sample

def remove_outliers(points, k=20, threshold=5.0, return_mask=False):
    """Remove statistical outliers using k-nearest neighbors statistics.
    
    Args:
        points: numpy array of shape (N, 3)
        k: number of nearest neighbors to consider
        threshold: threshold multiplier for outlier detection
        return_mask: if True, return (filtered_points, mask) instead of just filtered_points
    
    Returns:
        filtered_points if return_mask=False, else (filtered_points, mask) where mask is boolean array
    """
    if len(points) < k + 1:
        if return_mask:
            return points, np.ones(len(points), dtype=bool)
        return points
    
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    
    threshold_value = np.mean(mean_dist) + threshold * np.std(mean_dist)
    mask = mean_dist < threshold_value
    
    filtered_points = points[mask]
    if return_mask:
        return filtered_points, mask
    return filtered_points

def get_confidence_threshold_from_percentage(confidence, image_mask, mask_percentage=0.15):
    if hasattr(confidence, 'cpu'):
        confidence = confidence.cpu().detach().numpy()
    if hasattr(image_mask, 'cpu'):
        image_mask = image_mask.cpu().detach().numpy()
    
    image_mask = image_mask.astype(bool)
    
    confidence_flat = confidence.flatten()
    mask_flat = image_mask.flatten()
    
    foreground_conf = confidence_flat[mask_flat]
    

    percentile = mask_percentage * 100
    computed_threshold = np.percentile(foreground_conf, percentile)
    
    return computed_threshold

class AdHocDataset(Dataset):
    def __init__(self, cfg, ids=None, override_choices=None):
        """
        Args:
            cfg: Configuration object
            ids: List of subject IDs
            override_choices: Optional dict or single dict with override choices.
                If a single dict, it should contain:
                - 'subject_id': str, subject ID to override (e.g., '684')
                - 'scans': list, list of scan IDs (length should be num_frames_pp + 1 = 5)
                    First 4 scans are for main views, last 1 is for target pose
                - 'cameras': list, optional list of camera IDs (length should be num_frames_pp + 1 = 5)
                
                If a dict mapping index to override choices, each override choice dict can contain:
                - 'scans_main': list, list of scan IDs for main views (length should be num_frames_pp = 4)
                - 'scan_extra': str, scan ID for target pose
                - 'cameras': list, list of camera IDs (length should be num_frames_pp + 1 = 5)
        """
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

        # Handle simpler override format (single dict with subject_id)
        if override_choices is not None and 'subject_id' in override_choices:
            # Convert to index-based format for internal use
            self.override_choices = {}
            self.simple_override = override_choices
        else:
            self.override_choices = override_choices if override_choices is not None else {}
            self.simple_override = None

        

        self.crop_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
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
        return int(len(self.ids) * self.lengthen_by )
    
    def __getitem__(self, index):
        ret = defaultdict(list)
        ret['dataset'] = 'THuman'

        N, K = 4, 5 

        subject_id = self.ids[index // self.lengthen_by]
        gender = self.metadata[subject_id]['gender']
        ret['gender'] = gender

        scans_ids = self.metadata[subject_id]['scans']
        
        # Check for simple override format first
        overrides = {}
        if self.simple_override is not None and self.simple_override.get('subject_id') == subject_id:
            # Use simple override format
            simple = self.simple_override
            # Split scans: first 4 for main, last 1 for extra
            if 'scans' in simple:
                scans = simple['scans']
                if len(scans) != self.num_frames_pp + 1:
                    raise ValueError(f"scans must have length {self.num_frames_pp + 1}, got {len(scans)}")
                overrides['scans_main'] = scans[:self.num_frames_pp]
                overrides['scan_extra'] = scans[-1]
            
            # Handle cameras
            if 'cameras' in simple:
                overrides['cameras'] = simple['cameras']
        else:
            # Get override choices for this index if available (index-based format)
            overrides = self.override_choices.get(index, {})
        
        # Override scan sampling if provided
        # Support both 'scans_main'/'scan_extra' and 'frames_main'/'frame_extra' (for backward compatibility)
        if 'scans_main' in overrides and 'scan_extra' in overrides:
            sampled_scan_ids_main = overrides['scans_main']
            sampled_scan_extra = overrides['scan_extra']
        elif 'frames_main' in overrides and 'frame_extra' in overrides:
            # Backward compatibility: frames_main/frame_extra are actually scan IDs
            sampled_scan_ids_main = overrides['frames_main']
            sampled_scan_extra = overrides['frame_extra']
        else:
            sampled_scan_ids_main = None
            sampled_scan_extra = None
        
        if sampled_scan_ids_main is not None and sampled_scan_extra is not None:
            if len(sampled_scan_ids_main) != self.num_frames_pp:
                raise ValueError(f"scans_main/frames_main must have length {self.num_frames_pp}, got {len(sampled_scan_ids_main)}")
            # Validate main scans exist (these should be from the current subject)
            for scan in sampled_scan_ids_main:
                if scan not in scans_ids:
                    raise ValueError(f"Scan {scan} not found in scans for subject {subject_id}")
            # Note: scan_extra can be from a different subject (we only use its pose)
            # We don't validate it against scans_ids since it might be from another subject
            # The scan files will be loaded directly by scan_id, which should work if the scan exists in THuman
            sampled_scan_ids = list(sampled_scan_ids_main) + [sampled_scan_extra]
        else:
            sampled_scan_ids = np.random.choice(scans_ids, size=K, replace=True)
        ret['scan_ids'] = sampled_scan_ids
        
        # Override camera selection if provided
        if 'cameras' in overrides:
            sampled_cameras = overrides['cameras']
            if len(sampled_cameras) != self.num_frames_pp + 1:
                raise ValueError(f"cameras must have length {self.num_frames_pp + 1}, got {len(sampled_cameras)}")
            # Validate cameras exist
            for cam in sampled_cameras:
                if cam not in self.camera_ids:
                    raise ValueError(f"Camera {cam} not found in available camera_ids")
        else:
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
    


def visualise_and_save(batch, predictions, target_pose_scan, save_path):

    for k, v in predictions.items():
        predictions[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
    for k, v in batch.items():
        batch[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v

    B, N, H, W, C = predictions['vc_init'].shape
    K = 5
    
    image_masks = batch['masks']
    image_masks_N = image_masks[:, :N]

    smpl_masks = batch['smpl_mask']
    smpl_masks_N = smpl_masks[:, :N]

    mask_intersection = image_masks_N * smpl_masks_N
    mask_union = image_masks_N + smpl_masks_N - mask_intersection
    mask_union = mask_union.astype(bool)


    scatter_mask = image_masks_N[0].astype(bool) # nhw
    if "vc_init_conf" in predictions:
        confidence = predictions['vc_init_conf']
        # Compute threshold from percentage if enabled, otherwise use fixed threshold
        threshold_value = get_confidence_threshold_from_percentage(
            confidence[0], image_masks_N[0], mask_percentage=0.15
        )
        confidence = confidence > threshold_value
    else:
        confidence = np.ones_like(predictions['vc_init'])[..., 0].astype(bool)
    scatter_mask = scatter_mask * confidence[0].astype(bool)
    scatter_mask_flattened = rearrange(scatter_mask, 'n h w -> (n h w)')
    

    # Images are in [0, 1] range from ToTensor() transform (no normalization applied)
    # Need to scale to [0, 255] before converting to uint8
    images = rearrange(batch['imgs'][0, :4], 'n c h w -> (n h w) c')
    color = images[scatter_mask_flattened]
    # Scale from [0, 1] to [0, 255] and clamp to ensure valid range
    color = (color * 255.0).clip(0, 255)
    # Ensure color is uint8 and has 4 channels (RGBA)
    if color.shape[1] == 3:
        # Add alpha channel, set to 255 (fully opaque)
        alpha = np.full((color.shape[0], 1), 255, dtype=np.uint8)
        color = np.concatenate([color, alpha], axis=1)
    color = color.astype(np.uint8)


    gt_vp = batch['scan_verts'][0] # list of K point clouds
    gt_vp_faces = batch['scan_faces'][0]
    pred_vp = predictions['vp'][0] # k n h w 3
    pred_vp_init = predictions['vp_init'][0] # k n h w 3


    output_save_dir = save_path
    os.makedirs(output_save_dir, exist_ok=True)


    k = 4
    # Save GT point cloud (no colors)
    gt_points = gt_vp[k].cpu().detach().numpy()
    gt_faces = gt_vp_faces[k].cpu().detach().numpy()
    print(gt_points.shape)
    trimesh.Trimesh(
        vertices=gt_points,
        faces=gt_faces
    ).export(os.path.join(output_save_dir, f'gt_vp_{target_pose_scan}.ply'))
    # Save predicted vp point cloud (with colors)
    pred_points = pred_vp[k, scatter_mask]
    pred_points, outlier_mask = remove_outliers(pred_points, return_mask=True)
    # Filter colors to match the points after outlier removal
    color_filtered = color[outlier_mask]
    trimesh.PointCloud(
        vertices=pred_points,
        colors=color_filtered
    ).export(os.path.join(output_save_dir, f'pred_vp_{target_pose_scan}.ply'))
    
    # Save predicted vp_init point cloud (with colors)
    pred_init_points = pred_vp_init[k, scatter_mask]
    pred_init_points, outlier_mask_init = remove_outliers(pred_init_points, return_mask=True)
    # Filter colors to match the init points after outlier removal
    color_init_filtered = color[outlier_mask_init]
    trimesh.PointCloud(
        vertices=pred_init_points,
        colors=color_init_filtered
    ).export(os.path.join(output_save_dir, f'pred_vp_init_{target_pose_scan}.ply'))



if __name__ == '__main__':
    load_path = 'exp/exp_100_5_vp/saved_models/last.ckpt'
    cfg = get_cch_cfg_defaults()
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get config
    target_pose_scan = '1771'
    subject_ids = [str(i) for i in range(634, 681)]  # subject ids from 634 to 680 inclusive
    from core.data.thuman_metadata import THuman_metadata
    for subject_id in subject_ids:

        if len(THuman_metadata[subject_id]['scans']) < 4:
            print(f"Subject {subject_id} has less than 4 scans, skipping")
            continue
        scans_main = THuman_metadata[subject_id]['scans'][:4]
        vis_save_dir = f'Figures/THuman/THuman_id{subject_id}_pose{target_pose_scan}'
        model = CCHTrainer(
            cfg=cfg,
            dev=False,
            vis_save_dir=vis_save_dir
        )

        if load_path is not None:
            logger.info(f"Loading checkpoint: {load_path}")
            ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'], strict=True)
            
        model.eval()
        model.to(device)

        override_choices = {
            0: {
                'scans_main': scans_main,
                'scan_extra': target_pose_scan,
                'cameras': ['000', '090', '180', '270', '000']
            }
        }

        # Save the metadata as text in vis_save_dir (should match vis_save_dir given to CCHTrainer above)
        os.makedirs(vis_save_dir, exist_ok=True)
        metadata_path = os.path.join(vis_save_dir, f'metadata_{target_pose_scan}.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"scan_extra: {target_pose_scan}\n")
            for k, v in override_choices.items():
                f.write(f"Index: {k}\n")
                for key2, val2 in v.items():
                    f.write(f"  {key2}: {val2}\n")

        dataset = AdHocDataset(cfg, [subject_id, target_pose_scan], override_choices)   
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

        batch = next(iter(dataloader))
        batch = _move_to_device(batch, device)
        batch = model.process_thuman(batch)

        predictions = model(batch)

        model.visualiser.visualise(predictions, batch, split='val', epoch=0)

        visualise_and_save(batch, predictions, target_pose_scan, vis_save_dir)

        print('Done')

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


import sys
sys.path.append('.')

from core.models.trainer_4ddress import CCHTrainer
from core.configs.paths import BASE_PATH, DATA_PATH as PATH_TO_DATASET
from core.configs.model_size_cfg import MODEL_CONFIGS
from core.data.d4dress_utils import load_pickle, load_image, rotation_matrix, d4dress_cameras_to_pytorch3d_cameras
from core.configs.cch_cfg import get_cch_cfg_defaults
from core.data.full_dataset import custom_collate_fn


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


class AvatariserDataset(Dataset):
    def __init__(self, cfg, ids, override_choices=None):
        """
        Args:
            cfg: Configuration object
            ids: List of subject IDs
            override_choices: Optional dict or single dict with override choices.
                If a single dict, it should contain:
                - 'subject_id': str, subject ID to override (e.g., '00134')
                - 'take': str, take name to use (e.g., 'Take3')
                - 'frames': list, list of frame numbers (length should be num_frames_pp + 1 = 5)
                    First 4 frames are for main take, last 1 is for extra take
                - 'cameras': list, optional list of camera IDs (length should be num_frames_pp + 1 = 5)
                - 'layer': str, optional layer name ('Inner' or 'Outer'), defaults to 'Inner'
                
                If a dict mapping index to override choices, each override choice dict can contain:
                - 'take_index': int, index of take to use (instead of random)
                - 'take_name': tuple, (take_name, layer) to use (alternative to take_index)
                - 'extra_take_index': int, index of extra take to use
                - 'extra_take_name': tuple, (take_name, layer) to use (alternative to extra_take_index)
                - 'frames_main': list, list of frame numbers from main take (length should be num_frames_pp)
                - 'frame_extra': int, frame number from extra take
                - 'cameras': list, list of camera IDs (length should be num_frames_pp + 1)
        """
        self.cfg = cfg
        self.num_frames_pp = 4
        self.lengthen_by = cfg.DATA.LENGHTHEN_D4DRESS

        self.img_size = cfg.DATA.IMAGE_SIZE
        self.body_model = cfg.MODEL.BODY_MODEL
        self.num_joints = 24 if self.body_model == 'smpl' else 55

        self.subject_ids = ids 
        self.layer = ['Inner', 'Outer']
        self.camera_ids = ['0004', '0028', '0052', '0076']
        
        # Handle simpler override format (single dict with subject_id)
        if override_choices is not None and 'subject_id' in override_choices:
            # Convert to index-based format for internal use
            self.override_choices = {}
            self.simple_override = override_choices
        else:
            self.override_choices = override_choices if override_choices is not None else {}
            self.simple_override = None

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
            
            inner_takes = os.listdir(os.path.join(PATH_TO_DATASET, subject_id, self.layer[0]))
            inner_takes = [(take, 'Inner') for take in inner_takes if take.startswith('Take')]
            outer_takes = os.listdir(os.path.join(PATH_TO_DATASET, subject_id, self.layer[1]))
            outer_takes = [(take, 'Outer') for take in outer_takes if take.startswith('Take')]

            self.takes[subject_id] = inner_takes #+ outer_takes
            self.num_of_takes[subject_id] = len(inner_takes )#+ outer_takes)


        self.transform = transforms.Compose([
            transforms.CenterCrop((940, 940)),
            transforms.Resize((self.img_size, self.img_size)),
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
        ret = defaultdict(list)
        ret['dataset'] = '4DDress'

        subject_id = self.subject_ids[index // self.lengthen_by]
        
        # Check for simple override format first
        overrides = {}
        if self.simple_override is not None and self.simple_override.get('subject_id') == subject_id:
            # Use simple override format
            simple = self.simple_override
            take_name = simple.get('take')
            layer = simple.get('layer', 'Inner')  # Default to 'Inner' if not specified
            
            if take_name:
                # Find the take in the specified layer
                sampled_take = (take_name, layer)
                if sampled_take not in self.takes[subject_id]:
                    # Try to find in any layer
                    found = False
                    for l in self.layer:
                        if (take_name, l) in self.takes[subject_id]:
                            sampled_take = (take_name, l)
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Take {take_name} not found in any layer for subject {subject_id}")
                
                overrides['take_name'] = sampled_take
                # Use same take for extra take (can be overridden if needed)
                overrides['extra_take_name'] = sampled_take
                
                # Split frames: first 4 for main, last 1 for extra
                if 'frames' in simple:
                    frames = simple['frames']
                    if len(frames) != self.num_frames_pp + 1:
                        raise ValueError(f"frames must have length {self.num_frames_pp + 1}, got {len(frames)}")
                    overrides['frames_main'] = frames[:self.num_frames_pp]
                    overrides['frame_extra'] = frames[-1]
                
                # Handle cameras
                if 'cameras' in simple:
                    overrides['cameras'] = simple['cameras']
        else:
            # Get override choices for this index if available (index-based format)
            overrides = self.override_choices.get(index, {})

        num_of_takes = self.num_of_takes[subject_id]
        
        # Override take selection if provided
        if 'take_name' in overrides:
            take_name, layer = overrides['take_name']
            sampled_take = (take_name, layer)
            if sampled_take not in self.takes[subject_id]:
                raise ValueError(f"Take {sampled_take} not found in takes for subject {subject_id}")
        elif 'take_index' in overrides:
            sampled_take = self.takes[subject_id][overrides['take_index']]
        else:
            sampled_take = self.takes[subject_id][torch.randint(0, num_of_takes, (1,)).item()]
        
        take_dir = os.path.join(PATH_TO_DATASET, subject_id, sampled_take[1], sampled_take[0])

        # Sample an extra take in the same layer as sampled_take
        same_layer_takes = [t for t in self.takes[subject_id] if t[1] == sampled_take[1]]
        
        # Override extra take selection if provided
        if 'extra_take_name' in overrides:
            extra_take_name, extra_layer = overrides['extra_take_name']
            extra_take = (extra_take_name, extra_layer)
            if extra_take not in same_layer_takes:
                raise ValueError(f"Extra take {extra_take} not found in same layer takes for subject {subject_id}")
        elif 'extra_take_index' in overrides:
            # Find the index in same_layer_takes
            extra_take_idx = overrides['extra_take_index']
            if extra_take_idx >= len(same_layer_takes):
                raise ValueError(f"Extra take index {extra_take_idx} out of range for same layer takes")
            extra_take = same_layer_takes[extra_take_idx]
        else:
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
        
        # Override frame sampling if provided
        if 'frames_main' in overrides:
            sampled_frames_main = overrides['frames_main']
            if len(sampled_frames_main) != self.num_frames_pp:
                raise ValueError(f"frames_main must have length {self.num_frames_pp}, got {len(sampled_frames_main)}")
            # Validate frames exist
            for frame in sampled_frames_main:
                if frame not in scan_frames:
                    raise ValueError(f"Frame {frame} not found in scan_frames for main take")
        else:
            sampled_frames_main = np.random.choice(scan_frames, size=self.num_frames_pp, replace=False)
        
        if 'frame_extra' in overrides:
            sampled_frame_extra = overrides['frame_extra']
            if sampled_frame_extra not in extra_scan_frames:
                raise ValueError(f"Frame {sampled_frame_extra} not found in extra_scan_frames for extra take")
        else:
            sampled_frame_extra = np.random.choice(extra_scan_frames, size=1, replace=False)[0]
        
        sampled_frames = list(sampled_frames_main) + [sampled_frame_extra]
        
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
            sampled_cameras = self.camera_ids + [np.random.choice(self.camera_ids, size=1, replace=False).item()]
        
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



def visualise_and_save(batch, predictions, frame_extra, save_path):

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


    gt_vp = batch['scan_mesh_verts_centered'][0] # list of K point clouds
    gt_vp_faces = batch['scan_mesh_faces'][0]
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
    ).export(os.path.join(output_save_dir, f'gt_vp_{frame_extra}.ply'))
    # Save predicted vp point cloud (with colors)
    pred_points = pred_vp[k, scatter_mask]
    pred_points, outlier_mask = remove_outliers(pred_points, return_mask=True)
    # Filter colors to match the points after outlier removal
    color_filtered = color[outlier_mask]
    trimesh.PointCloud(
        vertices=pred_points,
        colors=color_filtered
    ).export(os.path.join(output_save_dir, f'pred_vp_{frame_extra}.ply'))
    
    # Save predicted vp_init point cloud (with colors)
    pred_init_points = pred_vp_init[k, scatter_mask]
    pred_init_points, outlier_mask_init = remove_outliers(pred_init_points, return_mask=True)
    # Filter colors to match the init points after outlier removal
    color_init_filtered = color[outlier_mask_init]
    trimesh.PointCloud(
        vertices=pred_init_points,
        colors=color_init_filtered
    ).export(os.path.join(output_save_dir, f'pred_vp_init_{frame_extra}.ply'))



if __name__ == '__main__':
    load_path = 'exp/exp_100_5_vp/saved_models/last.ckpt'
    save_path = 'exp/exp_single/'
    os.makedirs(save_path, exist_ok=True)
    cfg = get_cch_cfg_defaults()
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get config
    model = CCHTrainer(
        cfg=cfg,
        dev=False,
        vis_save_dir='Figures/vs_up2you/Take10_00117'
    )

    if load_path is not None:
        logger.info(f"Loading checkpoint: {load_path}")
        ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        
    model.eval()
    model.to(device)

    # frame_extra = '00088'
    # override_choices = {
    #     0: {
    #         'take_name': ('Take5', 'Inner'),
    #         'extra_take_name': ('Take5', 'Inner'),
    #         'frames_main': ['00072', '00091', '00068', '00070'],
    #         'frame_extra': frame_extra,
    #         'cameras': ['0004', '0028', '0052', '0076', '0076']
    #     }
    # }

    # frame_extra = '00079'
    # override_choices = {
    #     0: {
    #         'take_name': ('Take9', 'Inner'),
    #         'extra_take_name': ('Take5', 'Inner'),
    #         'frames_main': ['00024', '00104', '00013', '00129'],
    #         'frame_extra': frame_extra,
    #         'cameras': ['0004', '0028', '0052', '0076', '0076']
    #     }
    # }

    frame_extra = '00117'
    override_choices = {
        0: {
            'take_name': ('Take1', 'Inner'),
            'extra_take_name': ('Take10', 'Inner'),
            'frames_main': ['00072', '00091', '00068', '00070'],
            'frame_extra': frame_extra,
            'cameras': ['0004', '0028', '0052', '0076', '0076']
        }
    }

     # take 10 117 

    # Save the metadata as text in vis_save_dir
    vis_save_dir = 'Figures/vs_up2you/Take10_00117'  # Should match vis_save_dir given to CCHTrainer above
    os.makedirs(vis_save_dir, exist_ok=True)
    metadata_path = os.path.join(vis_save_dir, f'metadata_{frame_extra}.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"frame_extra: {frame_extra}\n")
        for k, v in override_choices.items():
            f.write(f"Index: {k}\n")
            for key2, val2 in v.items():
                f.write(f"  {key2}: {val2}\n")

    dataset = AvatariserDataset(cfg, ['00134'], override_choices)   
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    batch = next(iter(dataloader))
    batch = _move_to_device(batch, device)
    batch = model.process_4ddress(batch, 0)

    predictions = model(batch)

    model.visualiser.visualise(predictions, batch, split='val', epoch=0)

    visualise_and_save(batch, predictions, frame_extra, save_path)

    print('Done')

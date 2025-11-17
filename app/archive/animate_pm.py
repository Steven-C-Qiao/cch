"""
Given canonical pointmaps and skeleton, animate the mesh using pose sequences from CAPE

"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse 
import pickle

from einops import rearrange

from loguru import logger

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import look_at_view_transform

import matplotlib.pyplot as plt
viridis = plt.colormaps.get_cmap('viridis')

import sys 
sys.path.append('.')

from core.models.cch import CCH
from core.models.smpl import SMPL
from core.configs import paths
from core.configs.cch_cfg import get_cch_cfg_defaults
from core.utils.general_lbs import general_lbs
from core.utils.diffpointrend import PointCloudRenderer

MODEL_PATH = "/scratches/kyuban/cq244/CCH/cch/exp/exp_031_pred_w/saved_models/val_posed_loss_epoch=048.ckpt"
AMASS_POSE_PATH = '/scratches/kyuban/cq244/datasets/AMASS/CNRS/283/-01_L_1_stageii.npz'
CAPE_POSE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00134/longlong_ballet1_trial1'
AVATAR_PATH = '/scratches/kyuban/cq244/CCH/cch/tinkering/avatar.pkl'

def load_amass_pose_seq():
    pose_seq = np.load(AMASS_POSE_PATH, allow_pickle=True)

    for key, value in pose_seq.items():
        print(key, value.shape)

    import ipdb; ipdb.set_trace()

    pose_seq = pose_seq['pose']
    print(pose_seq.shape)
    pose_seq = torch.from_numpy(pose_seq).float()
    pose_seq = pose_seq.unsqueeze(0)
    print(pose_seq.shape)
    return pose_seq

def sample_cameras(num_views):
    # Create a single smooth rotation across all frames
    azim = torch.linspace(0, 360, num_views)
    elev = torch.zeros(num_views)  # Keep elevation constant
    dist = torch.ones(num_views) * 1.2  # Keep distance constant
    at = torch.tensor([0, -0.2, 0]).unsqueeze(0).repeat(num_views, 1)

    R, T = look_at_view_transform(
        dist=dist,
        elev=elev,
        azim=azim,
        at=at,
        degrees=True
    )
    return R, T

def load_cape_pose_seq():
    num_frames = 370
    pose_seq = []
    for i in range(1, num_frames+1):
        data = np.load(f'{CAPE_POSE_PATH}/longlong_ballet1_trial1.{i:06d}.npz')
        pose_seq.append(data['pose'])
    pose_seq = np.array(pose_seq)
    return pose_seq

def process_batch(vc_pred, w_pred, joints, pose_seq, smpl_model, device, batch_size=10):
    """Process frames in batches to avoid memory issues"""
    num_frames = pose_seq.shape[0]
    all_images = []
    
    # Sample cameras for the entire sequence once
    R, T = sample_cameras(num_views=num_frames)
    R = R.to(device)
    T = T.to(device)
    
    for i in range(0, num_frames, batch_size):
        end_idx = min(i + batch_size, num_frames)
        batch_pose = pose_seq[i:end_idx]
        batch_vc = vc_pred[i:end_idx]
        batch_w = w_pred[i:end_idx]
        batch_joints = joints[i:end_idx]
        
        # Process this batch
        vp_pred, joints_pred = general_lbs(
            vc=batch_vc,
            pose=batch_pose,
            lbs_weights=batch_w,
            J=batch_joints,
            parents=smpl_model.parents
        )
        
        # Create point clouds for this batch
        colors = torch.argmax(batch_w, dim=-1)
        colors_normalised = (colors / colors.max()).cpu().detach().numpy()
        colors = viridis(colors_normalised)[..., :3]
        colors = torch.tensor(colors).float().to(device)
        
        point_clouds = Pointclouds(
            points=vp_pred,
            features=colors
        )
        
        # Use the corresponding camera views for this batch
        batch_R = R[i:end_idx]
        batch_T = T[i:end_idx]
        
        # Render this batch
        batch_images = renderer(point_clouds, cam_R=batch_R, cam_t=batch_T)['images']
        all_images.append(batch_images.cpu().detach().numpy())
        
        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()
    
    return np.concatenate(all_images, axis=0)

def main(device, renderer):
    with torch.no_grad():
        conf_threshold = 0.05

        pose_seq = load_cape_pose_seq()
        pose_seq = torch.from_numpy(pose_seq).float().to(device)
        num_frames = pose_seq.shape[0]

        with open(AVATAR_PATH, 'rb') as f:
            avatar = pickle.load(f)

        vc_pred = avatar['vc_pred'][0]
        vc_conf = avatar['vc_conf'][0]
        w_pred = avatar['w_pred'][0]
        w_smpl = avatar['w_smpl'][0]
        joints = avatar['joints'][0]
        masks = avatar['masks'][0]
        conf_mask = (1/vc_conf) < conf_threshold

        full_mask = masks * conf_mask 
        # print(masks.shape, conf_mask.shape, full_mask.shape)

        # w = w_smpl # w_pred 
        w = w_pred

        vc_pred = rearrange(vc_pred, 'N H W C -> (N H W) C')
        vc_conf = rearrange(vc_conf, 'N H W -> (N H W)')
        w = rearrange(w, 'N H W J -> (N H W) J')
        full_mask = rearrange(full_mask, 'N H W -> (N H W)')

        vc_pred_masked = vc_pred[full_mask.astype(bool)]
        w_masked = w[full_mask.astype(bool)]

        # print(f'vc_pred_masked: {vc_pred_masked.shape}')
        # print(f'w_pred_masked: {w_pred_masked.shape}')  
        # print(f'pose_seq: {pose_seq.shape}')
        # print(f'joints: {joints.shape}')
        # print(f'full_mask: {full_mask.shape}')

        vc_pred = torch.tensor(vc_pred_masked).to(device)[None].repeat(num_frames, 1, 1)
        w = torch.tensor(w_masked).to(device)[None].repeat(num_frames, 1, 1)
        joints = torch.tensor(joints).to(device)[None].repeat(num_frames, 1, 1)

        # print(f'vc_pred: {vc_pred.shape}')
        # print(f'w_pred: {w_pred.shape}')
        # print(f'joints: {joints.shape}')

        # Process frames in batches
        images = process_batch(vc_pred, w, joints, pose_seq, smpl_model, device, batch_size=10)

        # Convert to numpy and scale to 0-255
        # set pixels to white only if all RGB channels are black
        black_mask = (images[..., 0] == 0) & (images[..., 1] == 0) & (images[..., 2] == 0)
        images[black_mask] = 1.0
        images = (images * 255.).astype(np.uint8)

        # Take every second frame
        images = images[::2]

        # save images as gif 
        import imageio
        imageio.mimsave('tinkering/output_pred_w.gif', images, fps=30)  # Adjusted fps for smoother playback with fewer frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir', 
        '-E', 
        type=str,
        help='Path to directory where logs and checkpoints are saved.'
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default='0,1', 
        help="Comma-separated list of GPU indices to use. E.g., '0,1,2'"
    )    
    parser.add_argument(
        "--dev", 
        action="store_true"
    )  
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    device_ids = list(map(int, args.gpus.split(",")))
    logger.info(f"Using GPUs: {args.gpus} (Device IDs: {device_ids})")

    smpl_model = SMPL(
        model_path=paths.SMPL,
        num_betas=10,
        gender='neutral'
    )

    renderer = PointCloudRenderer(
        device=device,
        batch_size=1,
        f=400,
        img_wh=512,
    )

    main(device=device, renderer=renderer)

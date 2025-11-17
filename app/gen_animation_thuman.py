import os
import torch
import pickle
import argparse
import trimesh
import smplx 
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence 

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import matrix_to_euler_angles, axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle


from tqdm import tqdm
from PIL import Image
from loguru import logger
from collections import defaultdict
from pytorch_lightning import seed_everything
from einops import rearrange
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append('.')

from core.configs.cch_cfg import get_cch_cfg_defaults
from core.data.thuman_dataset import THumanDataset, THuman_metadata, load_w_maps_sparse
from core.configs.paths import THUMAN_PATH
from core.models.trainer_4ddress import CCHTrainer
from core.data.d4dress_dataset import D4DressDataset
from core.data.full_dataset import custom_collate_fn
from core.configs.paths import DATA_PATH as PATH_TO_DATASET
from core.data.d4dress_utils import load_pickle, load_image, d4dress_cameras_to_pytorch3d_cameras
from core.utils.camera_utils import uv_to_pixel_space, custom_opengl2pytorch3d

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

# Process novel poses in smaller chunks to avoid OOM, and move results to CPU
def _move_to_cpu(sample):
    if isinstance(sample, torch.Tensor):
        return sample.detach().cpu()
    if isinstance(sample, dict):
        return {k: _move_to_cpu(v) for k, v in sample.items()}
    if isinstance(sample, list):
        return [_move_to_cpu(v) for v in sample]
    if isinstance(sample, tuple):
        return tuple(_move_to_cpu(v) for v in sample)
    return sample


def uniformly_sample_pointcloud(points, num_points=None, voxel_size=None):
    """
    Uniformly sample a point cloud in space using voxel grid downsampling.
    
    Args:
        points: torch.Tensor of shape (N, 3) - point cloud coordinates
        num_points: int - target number of points (if None, uses voxel_size)
        voxel_size: float - voxel size for downsampling (if None, auto-computed)
    
    Returns:
        torch.Tensor of shape (M, 3) - uniformly sampled point cloud
    """
    # Convert to numpy
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
        device = points.device
        dtype = points.dtype
    else:
        points_np = np.asarray(points)
        device = 'cpu'
        dtype = torch.float32
    
    # Check for empty or invalid point cloud
    if len(points_np) == 0:
        return torch.tensor(points_np, dtype=dtype, device=device)
    
    # Remove NaN and Inf values
    valid_mask = np.isfinite(points_np).all(axis=1)
    points_np = points_np[valid_mask]
    
    if len(points_np) == 0:
        return torch.tensor(points_np, dtype=dtype, device=device)
    
    # If num_points is specified, estimate voxel_size
    if num_points is not None and voxel_size is None:
        # Estimate voxel size based on bounding box and target number of points
        bbox_min = points_np.min(axis=0)
        bbox_max = points_np.max(axis=0)
        bbox_size = bbox_max - bbox_min
        volume = np.prod(bbox_size)
        if volume > 0 and num_points > 0:
            # Approximate voxel size to get roughly num_points
            voxel_size = np.cbrt(volume / num_points)
        else:
            voxel_size = 0.01  # default fallback
    
    # Perform voxel grid downsampling
    if voxel_size is not None and voxel_size > 0:
        # Compute voxel indices for each point
        bbox_min = points_np.min(axis=0)
        voxel_indices = np.floor((points_np - bbox_min) / voxel_size).astype(np.int64)
        
        # Use dictionary to keep one point per voxel (first point in each voxel)
        voxel_dict = {}
        for i, voxel_idx in enumerate(voxel_indices):
            voxel_key = tuple(voxel_idx)
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = i
        
        # Get sampled indices
        sampled_indices = np.array(list(voxel_dict.values()))
        points_sampled = points_np[sampled_indices]
    else:
        # If no downsampling specified, return original
        points_sampled = points_np
    
    # If num_points is specified and we have more points, randomly sample to exact number
    if num_points is not None and len(points_sampled) > num_points:
        indices = np.random.choice(len(points_sampled), num_points, replace=False)
        points_sampled = points_sampled[indices]
    elif num_points is not None and len(points_sampled) < num_points:
        # If we have fewer points, we can't upsample, so return what we have
        pass
    
    # Convert back to torch tensor
    return torch.tensor(points_sampled, dtype=dtype, device=device)


def get_novel_poses(pose_dir, smpl_model, device):
    pose_seq = []
    global_orient_seq = []
    for file in sorted(os.listdir(pose_dir)):
        if file.endswith('.pkl'):
            smpl_data = load_pickle(os.path.join(pose_dir, file))
            body_pose = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=device)[None]
            global_orient = torch.tensor(smpl_data['global_orient'], dtype=torch.float32, device=device)[None]
            betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=device)[None]
            transl = torch.tensor(smpl_data['transl'], dtype=torch.float32, device=device)[None]
            left_hand_pose = torch.tensor(smpl_data['left_hand_pose'], dtype=torch.float32, device=device)[None]
            right_hand_pose = torch.tensor(smpl_data['right_hand_pose'], dtype=torch.float32, device=device)[None]
            jaw_pose = torch.tensor(smpl_data['jaw_pose'], dtype=torch.float32, device=device)[None]
            leye_pose = torch.tensor(smpl_data['leye_pose'], dtype=torch.float32, device=device)[None]
            reye_pose = torch.tensor(smpl_data['reye_pose'], dtype=torch.float32, device=device)[None]
            expression = torch.tensor(smpl_data['expression'], dtype=torch.float32, device=device)[None]
            smpl_output = smpl_model(
                betas=betas,
                global_orient = global_orient,
                body_pose = body_pose,
                left_hand_pose = left_hand_pose,
                right_hand_pose = right_hand_pose,
                transl = transl,
                expression = expression,
                jaw_pose = jaw_pose,
                leye_pose = leye_pose,
                reye_pose = reye_pose,
                return_full_pose=True,
            )
            pose_seq.append(smpl_output.full_pose)
            global_orient_seq.append(global_orient)
    pose_seq = torch.stack(pose_seq)
    global_orient_seq = torch.stack(global_orient_seq)
    return pose_seq, global_orient_seq

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)




class AdHocDataset(THumanDataset):
    def __init__(
        self, cfg, 
        ids=None, 
        sampled_scan_ids=['2438', '2439', '2440', '2441', '2442'], 
        sampled_cameras=['000', '090', '180', '270', '000']
    ):
        super().__init__(cfg, ids)
        self.ids = ids
        self.sampled_scan_ids = sampled_scan_ids
        self.sampled_cameras = sampled_cameras


        self.crop_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )


    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        ret = defaultdict(list)
        ret['dataset'] = 'THuman'

        N, K = 4, 5 

        subject_id = self.ids[index]
        sampled_scan_ids = self.sampled_scan_ids
        
        ret['camera_ids'] = self.sampled_cameras

        for i, scan_id in enumerate(sampled_scan_ids):

            img_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, 'render', f'{self.sampled_cameras[i]}.png')
            rgba = Image.open(img_fname).convert('RGBA')
            
            alpha = np.array(rgba.split()[-1])
            mask = (alpha > 127).astype(np.uint8) * 255

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
            ret['scan_verts'].append(torch.tensor(scan['scan_verts']).float())
            ret['scan_faces'].append(torch.tensor(scan['scan_faces']).long())


            calib_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, 'calib', f'{self.sampled_cameras[i]}.txt')
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

            w_maps_dir_fname = os.path.join(THUMAN_PATH, 'render_persp/thuman2_36views', scan_id, f'{scan_id}_w_maps_sparse', f'{self.sampled_cameras[i]}.npz')
            w_map = load_w_maps_sparse(w_maps_dir_fname)  # Shape: (512, 512, 55)
            assert w_map.shape == (512, 512, 55)
            w_map = self.vc_map_transform(w_map)
            ret['smpl_w_maps'].append(w_map)

        for key in list(ret.keys()):
            if isinstance(ret[key], (list, tuple)) and len(ret[key]) > 0:
                try:
                    ret[key] = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in ret[key]])
                except:
                    pass
        return ret



class Solver:
    def __init__(self, subject_id, scan_ids, cameras, novel_pose_path, load_path, gt_scan_dir_path=None, gt_smpl_dir_path=None, save_dir=None):
        self.subject_id = subject_id
        self.scan_ids = scan_ids
        self.cameras = cameras
        self.stride = 5
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.novel_pose_path = novel_pose_path
        self.gt_scan_dir_path = gt_scan_dir_path
        self.gt_smpl_dir_path = gt_smpl_dir_path
        self.has_gt = (gt_scan_dir_path is not None) and (gt_smpl_dir_path is not None)
        seed_everything(42)

        if self.has_gt:
            self.gt_scan_fnames = sorted([
                fname for fname in os.listdir(gt_scan_dir_path)
                if fname.endswith('.pkl') and fname.startswith('mesh')
            ])
            self.gt_smpl_fnames = sorted([
                fname for fname in os.listdir(gt_smpl_dir_path)
                if fname.endswith('.pkl') and fname.startswith('mesh')
            ])
        else:
            self.gt_scan_fnames = []
            self.gt_smpl_fnames = []
            logger.info("Ground truth scans and SMPL fits not provided. Running in novel pose mode without ground truth comparison.")




        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f'Device: {device}')
        self.device = device

        # Get config
        cfg = get_cch_cfg_defaults()
        self.model = CCHTrainer(
            cfg=cfg,
            dev=False,
            vis_save_dir=None
        )
        self.smplx_neutral = self.model.smpl_neutral

        self.load_pretrained(model=self.model, load_path=load_path)

        self.dataset = AdHocDataset(cfg, ids=[subject_id], sampled_scan_ids=scan_ids, sampled_cameras=cameras)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

        
    def run(self):
        preds = defaultdict(list)

        init_batch = next(iter(self.dataloader))
        init_batch = _move_to_device(init_batch, self.device)
        images = init_batch['imgs']
        
        vc_ret = self.model.build_avatar_thuman(init_batch)
        confidence = vc_ret['vc_init_conf']




        def get_confidence_threshold_from_percentage(confidence, image_mask):
            """
            Compute threshold value that masks a certain percentage of foreground pixels with lowest confidence.
            
            Args:
                confidence: Confidence values tensor (B, N, H, W) or any shape
                image_mask: Foreground mask (B, N, H, W) or matching shape, can be boolean or numeric
                
            Returns:
                Threshold value to use for masking (scalar)
            """
            # Ensure mask is boolean tensor
            if not image_mask.dtype == torch.bool:
                image_mask = image_mask.bool()
            
            # Flatten for easier processing
            confidence_flat = confidence.flatten()
            mask_flat = image_mask.flatten()
            
            # Get confidence values only for foreground pixels
            foreground_conf = confidence_flat[mask_flat]
            
            if foreground_conf.numel() == 0:
                return 0.0
            
            # Calculate the threshold value for the given percentage
            # We want to mask the lowest mask_percentage of foreground pixels
            # Use quantile to get the threshold (equivalent to percentile)
            # quantile expects value in [0, 1] range, where 0.1 means 10th percentile
            computed_threshold = torch.quantile(foreground_conf.float(), 0.05)
            
            # Use the computed threshold
            return computed_threshold.item()

        conf_mask = confidence > get_confidence_threshold_from_percentage(confidence, init_batch['masks'][0, :4, ...])
        conf_mask = rearrange(conf_mask[0, :4, ...], 'n h w -> (n h w)').bool()

        for k, v in vc_ret.items():
            preds[k] = v
    
        novel_poses, global_orient_seq = get_novel_poses(self.novel_pose_path, self.smplx_neutral, self.device)


        
        vp_list = []

        
        distance_list = defaultdict(list)
        
        stride_indices = list(range(0, novel_poses.shape[0], self.stride))
        chunk_size = 10
        for start in tqdm(range(0, len(stride_indices), chunk_size)):
            end = min(start + chunk_size, len(stride_indices))
            for idx in stride_indices[start:end]:
                novel_pose = novel_poses[idx]

                vp = self.model.drive_avatar(vc_ret, init_batch, novel_pose)

                # Load ground truth data only if available
                if self.has_gt and idx < len(self.gt_smpl_fnames) and idx < len(self.gt_scan_fnames):
                    gt_smpl_data = load_pickle(os.path.join(self.gt_smpl_dir_path, self.gt_smpl_fnames[idx]))
                    gt_scan_transl = gt_smpl_data['transl']

                    gt_scan_mesh = pickle.load(open(os.path.join(self.gt_scan_dir_path, self.gt_scan_fnames[idx]), 'rb'))
                    gt_scan_verts = gt_scan_mesh['vertices']
                    gt_scan_verts_centered = gt_scan_verts - gt_scan_transl[None, :]
                    gt_scan_faces = gt_scan_mesh['faces']
                    gt_scan_mesh = trimesh.Trimesh(vertices=gt_scan_verts_centered, faces=gt_scan_faces)

                vp_cpu = _move_to_cpu(vp)
                vp_list.append(vp_cpu)


                mask = rearrange(init_batch['masks'][0, :4, ...], 'n h w -> (n h w)').bool()
                mask = mask * conf_mask

                pred_vp = rearrange(vp['vp'][0, -1, ...], 'n h w c -> (n h w) c')

                color = rearrange(images[0, :4, ...], 'n c h w -> (n h w) c')
                # Images are in [0, 1] range from ToTensor() transform (no normalization applied)
                # Need to scale to [0, 255] before converting to uint8
                color = color[mask]
                color = color.cpu().numpy()
                # Scale from [0, 1] to [0, 255] and clamp to ensure valid range
                color = (color * 255.0).clip(0, 255)
                # Ensure color is uint8 and has 4 channels (RGBA)
                if color.shape[1] == 3:
                    # Add alpha channel, set to 255 (fully opaque)
                    alpha = np.full((color.shape[0], 1), 255, dtype=np.uint8)
                    color = np.concatenate([color, alpha], axis=1)
                color = color.astype(np.uint8)


                pred_vp = pred_vp[mask]

                # Uniformly sample pred_vp in space before calculating chamfer distance
                # You can adjust num_points or voxel_size based on your needs
                # For example, to sample to ~10000 points: uniformly_sample_pointcloud(pred_vp, num_points=10000)
                # Or use a fixed voxel size: uniformly_sample_pointcloud(pred_vp, voxel_size=0.01)
                # pred_vp = uniformly_sample_pointcloud(pred_vp, voxel_size=0.01)

                # Save original pred_vp with colors before sampling
                if self.save_dir is not None:
                    save_path = os.path.join(self.save_dir, f"pred_vp_{idx:03d}.ply")
                    trimesh.PointCloud(pred_vp.cpu().numpy(), colors=color).export(save_path)
                    # Only save ground truth mesh if available
                    if self.has_gt and idx < len(self.gt_scan_fnames):
                        save_path = os.path.join(self.save_dir, f"gt_vp_{idx:03d}.ply")
                        gt_scan_mesh.export(save_path)

                # Calculate chamfer distance only if ground truth is available
                if self.has_gt and idx < len(self.gt_smpl_fnames) and idx < len(self.gt_scan_fnames):
                    pred_vp_for_cfd = Pointclouds(points=pred_vp[None])

                    print(pred_vp_for_cfd.points_packed().shape)

                    gt_vp = Pointclouds(points=torch.tensor(gt_scan_verts_centered[None], dtype=torch.float32, device=self.device))

                    cfd_ret, _ = chamfer_distance(
                        pred_vp_for_cfd,
                        gt_vp,
                        point_reduction=None,
                        batch_reduction=None,
                    )

                    cfd_sqrd_pred2gt = cfd_ret[0]  
                    cfd_sqrd_gt2pred = cfd_ret[1]

                    cfd_pred2gt = torch.sqrt(cfd_sqrd_pred2gt).mean() * 100.0
                    cfd_gt2pred = torch.sqrt(cfd_sqrd_gt2pred).mean() * 100.0

                    print(cfd_pred2gt, cfd_gt2pred)

                    distance_list['cfd_pred2gt'].append(cfd_pred2gt.item())
                    distance_list['cfd_gt2pred'].append(cfd_gt2pred.item())
                    distance_list['cfd'].append((cfd_pred2gt.item() + cfd_gt2pred.item())/2)

                del vp, vp_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Only print distance statistics if ground truth was available
        if self.has_gt and len(distance_list) > 0:
            for k, v in distance_list.items():
                distance_list[k] = np.mean(v)
                print(f'{k}: {distance_list[k]}')

        preds['vp_list'] = vp_list

        # self.visualise(preds, init_batch)

        # Save predictions and batch data
        logger.info("Saving predictions and batch data...")
        
        return vp_list
    


    def load_pretrained(self, model, load_path=None):
        if load_path is not None:
            logger.info(f"Loading checkpoint: {load_path}")
            ckpt = torch.load(load_path, weights_only=False, map_location='cpu')
            model_state = model.state_dict()
            pretrained_state = ckpt['state_dict']
            
            # Print shape mismatches
            mismatched = {k: (v.shape, model_state[k].shape) 
                        for k, v in pretrained_state.items() 
                        if k in model_state and v.shape != model_state[k].shape}
            if mismatched:
                logger.info("Shape mismatches found:")
                for k, (pretrained_shape, model_shape) in mismatched.items():
                    logger.info(f"{k}: checkpoint shape {pretrained_shape}, model shape {model_shape}")
            
            filtered_state = {k: v for k, v in pretrained_state.items() 
                            if k in model_state and v.shape == model_state[k].shape}
            logger.info(f"Loading {len(filtered_state)}/{len(pretrained_state)} keys from checkpoint")
            model.load_state_dict(filtered_state, strict=True)
            
        model.eval()
        model.to(device)
        return model 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--load_from_ckpt', 
        '-L', 
        type=str, 
        default=None,
        help='Path to checkpoint. Load for finetuning'
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default='0', 
        help="Comma-separated list of GPU indices to use. E.g., '0,1,2'"
    )    
    parser.add_argument(
        "--dev", 
        action="store_true"
    )  
    args = parser.parse_args()
    assert (args.load_from_ckpt is not None), 'Specify load_from_ckpt'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    cfg = get_cch_cfg_defaults()
    cfg.TRAIN.BATCH_SIZE = 1

    

    smplx_male = smplx.create(
        model_type="smplx",
        model_path="model_files/",
        num_pca_comps=12,
    ).to(device)


    ''' ---------------------------------------------------------- '''
    subject_id = '2417'
    scan_ids = ['2417', '2417', '2417', '2417', '2417']
    cameras = ['000', '090', '180', '270', '000']
    novel_pose_path = f'/scratch/u5au/chexuan.u5au/4DDress/00148/Inner/Take1/SMPLX'
    gt_scan_path = f'/scratch/u5au/chexuan.u5au/4DDress/00148/Inner/Take1/Meshes_pkl'
    take='Take1'



    novel_pose_path = f'/scratch/u5au/chexuan.u5au/4DDress/00134/Inner/{take}/SMPLX'
    gt_scan_dir_path = None
    gt_smpl_dir_path = None
    save_dir = f'Figures/vis/THuman_{subject_id}_00134_Take1'
    os.makedirs(save_dir, exist_ok=True)

    solver = Solver(
        subject_id, scan_ids, cameras, novel_pose_path, args.load_from_ckpt, gt_scan_dir_path, gt_smpl_dir_path, save_dir)
    solver.run()





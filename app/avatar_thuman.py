import os
import torch
import pickle
import argparse
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence 



from tqdm import tqdm
from PIL import Image
from loguru import logger
from collections import defaultdict
from pytorch_lightning import seed_everything
from einops import rearrange
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from pytorch3d.transforms import matrix_to_euler_angles, axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle

import sys
sys.path.append('.')

from core.configs.cch_cfg import get_cch_cfg_defaults
from core.data.thuman_dataset import THumanDataset, THuman_metadata, load_w_maps_sparse
from core.data.full_dataset import custom_collate_fn as d4dress_collate_fn
from core.data.d4dress_utils import load_pickle
from core.models.trainer_4ddress import CCHTrainer
from core.configs.paths import THUMAN_PATH
from core.utils.camera_utils import uv_to_pixel_space, custom_opengl2pytorch3d
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _no_annotations(fig):
    for ax in fig.axes:
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none') 
        ax.zaxis.pane.set_edgecolor('none')
        ax.xaxis.line.set_color('none')
        ax.yaxis.line.set_color('none')
        ax.zaxis.line.set_color('none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


def _set_scatter_limits(ax, x, elev=10, azim=90):
    max_range = np.array([
        x[:, 0].max() - x[:, 0].min(),
        x[:, 1].max() - x[:, 1].min(),
        x[:, 2].max() - x[:, 2].min()
    ]).max() / 2.0 + 0.05
    mid_x = (x[:, 0].max() + x[:, 0].min()) * 0.5
    mid_y = (x[:, 1].max() + x[:, 1].min()) * 0.5
    mid_z = (x[:, 2].max() + x[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5)) 
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))


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


def _get_confidence_threshold_from_percentage(confidence, image_mask, mask_percentage=0.0):
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
        gender = self.metadata[subject_id]['gender']
        ret['gender'] = gender

        scans_ids = self.metadata[subject_id]['scans']
        sampled_scan_ids = self.sampled_scan_ids
        ret['scan_ids'] = sampled_scan_ids
        
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
    def __init__(self, subject_id, scan_ids, cameras, novel_pose_path, load_path, save_name):
        self.subject_id = subject_id
        self.scan_ids = scan_ids
        self.cameras = cameras
        self.novel_pose_path = novel_pose_path
        self.save_name = save_name

        self.stride = 2

        seed_everything(42)

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
        self.smplx_male = self.model.smpl_male
        self.smplx_female = self.model.smpl_female

        self.load_pretrained(model=self.model, load_path=load_path)

        self.dataset = AdHocDataset(cfg, ids=[subject_id], sampled_scan_ids=scan_ids, sampled_cameras=cameras)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=d4dress_collate_fn)

        
    def run(self):
        preds = defaultdict(list)

        init_batch = next(iter(self.dataloader))
        init_batch = _move_to_device(init_batch, self.device)
        
        vc_ret = self.model.build_avatar_thuman(init_batch)
        for k, v in vc_ret.items():
            preds[k] = v
    
        novel_poses = self.get_novel_poses(self.novel_pose_path)


        
        vp_list = []
        
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

        stride_indices = list(range(0, novel_poses.shape[0], self.stride))
        chunk_size = 10
        for start in tqdm(range(0, len(stride_indices), chunk_size)):
            end = min(start + chunk_size, len(stride_indices))
            for idx in stride_indices[start:end]:
                novel_pose = novel_poses[idx]
                vp = self.model.drive_avatar(vc_ret, init_batch, novel_pose)
                vp_cpu = _move_to_cpu(vp)
                vp_list.append(vp_cpu)
                del vp, vp_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        preds['vp_list'] = vp_list

        self.visualise(preds, init_batch)

        return vp_list
    


    def get_novel_poses(self, pose_dir):
        pose_seq = []
        for file in sorted(os.listdir(pose_dir)):
            if file.endswith('.pkl'):
                smpl_data = load_pickle(os.path.join(pose_dir, file))
                body_pose = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=self.device)[None]
                global_orient = torch.tensor(smpl_data['global_orient'], dtype=torch.float32, device=self.device)[None]
                betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=self.device)[None]
                transl = torch.tensor(smpl_data['transl'], dtype=torch.float32, device=self.device)[None]
                left_hand_pose = torch.tensor(smpl_data['left_hand_pose'], dtype=torch.float32, device=self.device)[None]
                right_hand_pose = torch.tensor(smpl_data['right_hand_pose'], dtype=torch.float32, device=self.device)[None]
                jaw_pose = torch.tensor(smpl_data['jaw_pose'], dtype=torch.float32, device=self.device)[None]
                leye_pose = torch.tensor(smpl_data['leye_pose'], dtype=torch.float32, device=self.device)[None]
                reye_pose = torch.tensor(smpl_data['reye_pose'], dtype=torch.float32, device=self.device)[None]
                expression = torch.tensor(smpl_data['expression'], dtype=torch.float32, device=self.device)[None]
                smpl_output = self.smplx_male(
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
        pose_seq = torch.stack(pose_seq)
        return pose_seq
        

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



    def visualise(self, predictions, batch):
        for k, v in predictions.items():
            predictions[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items():
            batch[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v

        subfig_size = 8
        s = 0.05 # scatter point size
        gt_alpha, pred_alpha = 0.5, 0.5

        B, N, H, W, C = predictions['vc_init'].shape
        K = 5

        mask = batch['masks']
        mask_N = mask[:, :N]

        num_cols = 4
        num_total_plots = len(predictions['vp_list'])
        num_rows = num_total_plots // num_cols + (num_total_plots % num_cols > 0)


        scatter_mask = mask_N[0].astype(bool) # nhw
        if "vc_init_conf" in predictions:
            confidence = predictions['vc_init_conf']
            # Compute threshold from percentage if enabled, otherwise use fixed threshold
            threshold_value = _get_confidence_threshold_from_percentage(
                confidence[0], mask_N[0], mask_percentage=0.01
            )
            confidence = confidence > threshold_value
        else:
            confidence = np.ones_like(predictions['vc_init'])[..., 0].astype(bool)
        scatter_mask = scatter_mask * confidence[0].astype(bool)
        # Color for predicted scatters 
        color = rearrange(batch['imgs'][0, :N], 'n c h w -> n h w c')
        # color = np.zeros_like(color)
        # base_colors = [
        #     np.array([1.0, 0.0, 0.0]),  # red
        #     np.array([0.0, 1.0, 0.0]),  # green
        #     np.array([0.0, 0.0, 1.0]),  # blue
        #     np.array([1.0, 0.647, 0.0]), # orange (RGB)
        # ]
        # color[0] = base_colors[0]
        # color[1] = base_colors[1]
        # color[2] = base_colors[2]
        # color[3] = base_colors[3]
        color = color[scatter_mask]
        color = color.astype(np.float32)


        x = batch['smpl_T_joints'].reshape(-1, 3)

        self.x = x

        # self.gen_gt_animation(gt_scan_path)

        # Save individual subplots for gif
        frames = []
        temp_fig = plt.figure(figsize=(subfig_size*4, subfig_size))
        azims = [0, 90, 180, 270]
        
        for i in range(num_total_plots):
            verts = predictions['vp_list'][i]['vp'][0, -1, ...].cpu().detach().numpy()
            verts = verts[scatter_mask]
            
            for j, azim in enumerate(azims):
                ax = temp_fig.add_subplot(1, 4, j+1, projection='3d')
                ax.scatter(
                    verts[:, 0], 
                    verts[:, 1], 
                    verts[:, 2], c=color, s=s, alpha=gt_alpha
                )
                # ax.set_title(f'pred $V^{{{i+1}}}$')
                _set_scatter_limits(ax, x, elev=10, azim=azim)
            _no_annotations(temp_fig)
            plt.tight_layout(pad=0.)
            
            # Save frame to memory
            temp_fig.canvas.draw()
            frame = np.frombuffer(temp_fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(temp_fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(Image.fromarray(frame))
            
            # Clear figure for next frame
            plt.clf()
        
        plt.close(temp_fig)
        
        # Save as gif
        frames[0].save(
            f'{self.save_name}.gif',
            save_all=True,
            append_images=frames[1:],
            duration=100, # 500ms per frame
            loop=0,
            dpi=(200, 200) # Higher DPI for better quality
        )

        # Create GIF for vp_init if available
        vp_init_frames = []
        # Create a new figure for vp_init
        vp_init_fig = plt.figure(figsize=(subfig_size*4, subfig_size))
        
        for i in range(num_total_plots):  # Iterate over K frames
            verts = predictions['vp_list'][i]['vp_init'][0, -1, ...].cpu().detach().numpy()
            verts = verts[scatter_mask]
            
            for j, azim in enumerate(azims):
                ax = vp_init_fig.add_subplot(1, 4, j+1, projection='3d')
                ax.scatter(
                    verts[:, 0], 
                    verts[:, 1], 
                    verts[:, 2], c=color, s=s, alpha=gt_alpha
                )
                _set_scatter_limits(ax, x, elev=10, azim=azim)
            _no_annotations(vp_init_fig)
            plt.tight_layout(pad=0.)
            
            # Save frame to memory
            vp_init_fig.canvas.draw()
            frame = np.frombuffer(vp_init_fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(vp_init_fig.canvas.get_width_height()[::-1] + (3,))
            vp_init_frames.append(Image.fromarray(frame))
            
            # Clear figure for next frame
            plt.clf()
        
        plt.close(vp_init_fig)
            
        # Save vp_init as gif
        vp_init_frames[0].save(
            f'{self.save_name}_vp_init.gif',
            save_all=True,
            append_images=vp_init_frames[1:],
            duration=100, # 500ms per frame
            loop=0,
            dpi=(200, 200) # Higher DPI for better quality
        )


    def gen_gt_animation(self, path):

        subfig_size = 8
        s = 0.05 # scatter point size
        gt_alpha = 0.5
        gt_frames = []
        azims = [0, 90, 180, 270]
        gt_fig = plt.figure(figsize=(4*4, 4))

        mesh_fnames = sorted(f for f in os.listdir(path) if (f.endswith('.pkl') and f.startswith('mesh-f')))
        x = None
        for mesh_fname in mesh_fnames[::self.stride]:
            mesh = load_pickle(os.path.join(path, mesh_fname))

            gt_verts = mesh['vertices']

            if x is None:
                x = gt_verts
            gt_color = mesh['colors']

            for j, azim in enumerate(azims):
                ax = gt_fig.add_subplot(1, 4, j+1, projection='3d')
                ax.scatter(
                    gt_verts[:, 0], 
                    gt_verts[:, 1], 
                    gt_verts[:, 2], s=s, alpha=gt_alpha#, c=gt_color
                )
                _set_scatter_limits(ax, x, elev=10, azim=azim)
            _no_annotations(gt_fig)
            plt.tight_layout(pad=0.)

            # Save frame to memory
            gt_fig.canvas.draw()
            frame = np.frombuffer(gt_fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(gt_fig.canvas.get_width_height()[::-1] + (3,))
            gt_frames.append(Image.fromarray(frame))
            
            # Clear figure for next frame
            plt.clf()

        plt.close(gt_fig)

        gt_frames[0].save(
            f'{self.save_name}_gt.gif',
            save_all=True,
            append_images=gt_frames[1:],
            duration=100, # 500ms per frame
            loop=0,
            dpi=(200, 200) # Higher DPI for better quality
        )
        return gt_frames

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

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


    assert (args.load_from_ckpt is not None), 'Specify load_from_ckpt'


    
    subject_id = '684'
    scan_ids = ['2438', '2439', '2440', '2441', '2442']
    cameras = ['000', '090', '180', '270', '000']
    novel_pose_path = f'/scratch/u5au/chexuan.u5au/4DDress/00148/Inner/Take1/SMPLX'
    gt_scan_path = f'/scratch/u5au/chexuan.u5au/4DDress/00148/Inner/Take1/Meshes_pkl'

    save_name = f'Animations/THuman_{subject_id}'


    solver = Solver(subject_id, scan_ids, cameras, novel_pose_path, args.load_from_ckpt, save_name)
    solver.run()

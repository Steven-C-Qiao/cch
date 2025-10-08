import os
import torch
import pickle
import argparse
import trimesh
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm
from PIL import Image
from loguru import logger
from collections import defaultdict
from pytorch_lightning import seed_everything
from einops import rearrange
from torch.utils.data import DataLoader

import sys
sys.path.append('.')

from core.configs.cch_cfg import get_cch_cfg_defaults
from core.models.trainer_4ddress import CCHTrainer
from core.data.d4dress_dataset import D4DressDataset, d4dress_collate_fn
from core.configs.paths import DATA_PATH as PATH_TO_DATASET
from core.data.d4dress_utils import load_pickle, load_image, d4dress_cameras_to_pytorch3d_cameras


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

class AdHocDataset(D4DressDataset):
    def __init__(self, cfg, ids=[], layer='Inner', takes=['Take3'],
                 frames=['00011', '00011', '00011', '00011', '00015'],
                 cameras=['0004', '0028', '0052', '0076', '0076']):
        super().__init__(cfg, ids)
        self.ids = ids
        self.layer = layer
        self.takes = takes
        self.frames = frames
        self.cameras = cameras
        

    def __getitem__(self, index):
        ret = defaultdict(list)

        id = self.ids[index]
        layer = self.layer 
        take_dir = os.path.join(PATH_TO_DATASET, id, layer, self.takes[index])
        sampled_frames = self.frames
        sampled_cameras = self.cameras

        ret['take_dir'] = take_dir

        basic_info = load_pickle(os.path.join(take_dir, 'basic_info.pkl'))
        gender = basic_info['gender'] # is str
        ret['gender'] = gender
        scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']
        
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
            
            # Convert to PIL Images for transforms
            img_pil = Image.fromarray(img)
            mask_pil = Image.fromarray(mask)
            
            # Apply transforms
            # img_transformed = self.normalise(self.transform(img_pil))
            img_transformed = self.transform(img_pil)
            mask_transformed = self.mask_transform(mask_pil).squeeze()

            # sapiens_image = sapiens_transform(img_pil)

            # ret['sapiens_images'].append(sapiens_image)
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
        # ret['sapiens_images'] = torch.tensor(np.stack(ret['sapiens_images']))
        # ret['smpl_w_maps'] = torch.stack(ret['smpl_w_maps'])
        if self.body_model == 'smplx':
            ret['left_hand_pose'] = torch.tensor(np.stack(ret['left_hand_pose']))
            ret['right_hand_pose'] = torch.tensor(np.stack(ret['right_hand_pose']))
            ret['jaw_pose'] = torch.tensor(np.stack(ret['jaw_pose']))
            ret['leye_pose'] = torch.tensor(np.stack(ret['leye_pose']))
            ret['reye_pose'] = torch.tensor(np.stack(ret['reye_pose']))
            ret['expression'] = torch.tensor(np.stack(ret['expression']))

        return ret




class Solver:
    def __init__(self, id, take, frames, cameras, novel_pose_path, load_path):
        self.id = id
        self.take = take
        self.novel_pose_path = novel_pose_path

        self.stride = 25

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

        self.dataset = AdHocDataset(cfg, ids=[id], takes=[take], frames=frames, cameras=cameras)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=d4dress_collate_fn)

        
    def run(self):
        preds = defaultdict(list)

        init_batch = next(iter(self.dataloader))
        init_batch = _move_to_device(init_batch, self.device)
        
        vc_ret = self.model.build_avatar(init_batch)
        for k, v in vc_ret.items():
            preds[k] = v
    
        novel_poses = self.get_novel_poses(novel_pose_path)
        
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

        # Save predictions and batch data
        logger.info("Saving predictions and batch data...")
        
        # Create output directory if it doesn't exist
        os.makedirs('app/avatar_outputs', exist_ok=True)
        
        # Save vp_list
        with open('app/avatar_outputs/vp_list.pkl', 'wb') as f:
            pickle.dump(vp_list, f)
            
        # Save full predictions dict
        with open('app/avatar_outputs/preds.pkl', 'wb') as f:
            pickle.dump(preds, f)
            
        # Save initial batch
        with open('app/avatar_outputs/init_batch.pkl', 'wb') as f:
            pickle.dump(init_batch, f)

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
            confidence = confidence > 15.
        else:
            confidence = np.ones_like(predictions['vc_init'])[..., 0].astype(bool)
        scatter_mask = scatter_mask * confidence[0].astype(bool)
        # Color for predicted scatters 
        color = rearrange(batch['imgs'][0, :N], 'n c h w -> n h w c')
        color = color[scatter_mask]
        color = color.astype(np.float32)


        x = batch['smpl_T_joints'].reshape(-1, 3)
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
            f'{self.id}_{self.take}_animation.gif',
            save_all=True,
            append_images=frames[1:],
            duration=100, # 500ms per frame
            loop=0,
            dpi=(200, 200) # Higher DPI for better quality
        )


            
        fig = plt.figure(figsize=(subfig_size * num_cols, subfig_size * num_rows))
        r = 0


        for i in range(num_total_plots):
            verts = predictions['vp_list'][i]['vp'][0, -1, ...].cpu().detach().numpy() # B K N H W 3 -> N H W 3
            verts = verts[scatter_mask]
            ax = fig.add_subplot(num_rows, num_cols, i+1, projection='3d')
            ax.scatter(verts[:, 0], 
                    verts[:, 1], 
                    verts[:, 2], c=color, s=s, alpha=gt_alpha)
            ax.set_title(f'pred $V^{{{i+1}}}$')
            _set_scatter_limits(ax, x)

        plt.tight_layout()
        plt.savefig(f'{self.id}_{self.take}_animation.png', dpi=200)
        plt.close()


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

    id = '00134'
    take = 'Take3'
    frames = ['00006', '00006', '00006', '00006', '00006']
    cameras = ['0004', '0028', '0052', '0076', '0076']
    novel_pose_path = f'/scratch/cq244/datasets/4DDress/{id}/Inner/{take}/SMPLX'


    # id = '00147'
    # take = 'Take6'
    # frames = ['00021', '00021', '00021', '00021', '00021']
    # cameras = ['0004', '0028', '0052', '0076', '0076']
    # novel_pose_path = f'/scratch/cq244/datasets/4DDress/{id}/Inner/{take}/SMPLX'


    # id = '00148'
    # take = 'Take1'
    # frames = ['00021', '00021', '00021', '00021', '00021']
    # cameras = ['0004', '0028', '0052', '0076', '0076']
    # novel_pose_path = f'/scratch/cq244/datasets/4DDress/{id}/Inner/{take}/SMPLX'


    # id = '00188'
    # take = 'Take1'
    # frames = ['00021', '00021', '00021', '00021', '00021']
    # cameras = ['0004', '0028', '0052', '0076', '0076']
    # novel_pose_path = f'/scratch/cq244/datasets/4DDress/{id}/Inner/{take}/SMPLX'



    solver = Solver(id, take, frames, cameras, novel_pose_path, args.load_from_ckpt)
    solver.run()

import os
import torch
import argparse
from collections import defaultdict
from loguru import logger
import trimesh
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

import sys
sys.path.append('.')

from core.configs.cch_cfg import get_cch_cfg_defaults
from core.models.trainer_4ddress import CCHTrainer
from core.data.d4dress_datamodule import CCHDataModule
from core.data.d4dress_dataset import D4DressDataset
from core.configs.paths import DATA_PATH as PATH_TO_DATASET
from core.data.d4dress_utils import load_pickle, load_image, d4dress_cameras_to_pytorch3d_cameras

class AdHocDataset(D4DressDataset):
    

    def set_images(
        self, 
        id='00122',
        take_dir='/scratch/u5aa/chexuan.u5aa/4DDress/00122/Inner/Take2', 
        frames=['00011', '00012', '00013', '00014', '00015'],
        cameras=['0004', '0028', '0052', '0076', '0076']
    ):
        ret = defaultdict(list)
        ret['take_dir'] = take_dir
        ret['frames'] = frames
        ret['cameras'] = cameras
        ret['id'] = id
        return ret 


    def __getitem__(self, index):
        super().__getitem__(index)
        ret = defaultdict(list)

        sample_ret = self.set_images()
        id = sample_ret['id']
        layer = self.layer 
        take_dir = sample_ret['take_dir']
        sampled_frames = sample_ret['frames']
        sampled_cameras = sample_ret['cameras']

        ret['take_dir'] = sample_ret['take_dir']

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



def run_test(dev=False, load_path=None):
    seed_everything(42)
    
    # Get config
    cfg = get_cch_cfg_defaults()

    model = CCHTrainer(
        cfg=cfg,
        dev=dev,
        vis_save_dir=None
    )

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



    dataset = AdHocDataset(cfg, ids=['00122'])

    init_batch = dataset[0]
    
    vc_ret = model.build_avatar(init_batch)


    def get_novel_poses(pose_dir):
        pose_seq = []
        for file in os.listdir(pose_dir):
            if file.endswith('.pkl'):
                pose_seq.append(load_pickle(os.path.join(pose_dir, file))['pose'])
        return pose_seq

    novel_poses = get_novel_poses('/scratches/kyuban/cq244/datasets/4DDress/00122/Inner/Take2/SMPLX')
    print(len(novel_poses))
    
    vp_list = []
    for novel_pose in novel_poses[::5]:
        novel_pose = torch.from_numpy(novel_pose).float().to(device)
        vp = model.drive_avatar(vc_ret, init_batch, novel_pose)

        vp_list.append(vp)

    import ipdb; ipdb.set_trace()
    return vp_list
        

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
        default=None, 
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


    assert (args.load_from_ckpt is not None), 'Specify load_from_ckpt'

    run_test(
        dev=args.dev,
        load_path=args.load_from_ckpt
    )

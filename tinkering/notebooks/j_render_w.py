import torch 
import os
import smplx 
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import scipy

from torch.utils.data import DataLoader
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import TexturesVertex
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points

from core.configs.paths import BASE_PATH, DATA_PATH as PATH_TO_DATASET
from core.data.d4dress_dataset import D4DressDataset
from core.data.d4dress_utils import load_pickle, d4dress_cameras_to_pytorch3d_cameras
from core.utils.feature_renderer import FeatureRenderer

import time 


ids = [
    '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137', 
    '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156', 
    '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175', 
    '00176', '00179', '00180', '00185', '00187', '00190', '00188', '00191'
] 

layer = 'Inner'
# camera_ids = ['0004', '0028', '0052', '0076']


smpl_male = smplx.create(
    model_type='smplx',
    model_path="model_files/",
    num_betas=10,
    gender='male',
    num_pca_comps=12,
)
smpl_female = smplx.create(
    model_type='smplx',
    model_path="model_files/",
    num_betas=10,
    gender='female',
    num_pca_comps=12,
)

class Solver:
    def __init__(self):
        self.ids = ids
        self.layer = layer
        self.device = 'cuda'

        self.renderer = FeatureRenderer(image_size=(320, 235)).to(self.device)

        self.smpl_male = smpl_male.to(self.device)
        self.smpl_female = smpl_female.to(self.device)

        # self.dataset = D4DressDataset(cfg=self.cfg, ids=self.ids)
        # self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def solve_all(self):
        start_time = time.time()
        for id in self.ids:
            time_id_start = time.time()
            self.solve_id(id)
            time_id_end = time.time()
            print(f"Time taken for id {id}: {time_id_end - time_id_start} seconds")
        end_time = time.time()
        print(f"Total time taken: {end_time - start_time} seconds")
        return None

    def solve_id(self, id):
        takes = os.listdir(os.path.join(PATH_TO_DATASET, id, self.layer))
        takes = [take for take in takes if take.startswith('Take')]

        for take in takes:
            take_dir = os.path.join(PATH_TO_DATASET, id, layer, take)
            print(f"Processing {take_dir}")

            
            self.solve_take(id, take_dir)

        return None 
    
    def solve_take(self, id, take_dir):
        camera_params = load_pickle(os.path.join(take_dir, 'Capture', 'cameras.pkl'))
        camera_ids = list(camera_params.keys())
        
        for camera in camera_ids:
            save_dir = os.path.join(take_dir, 'Capture', camera, 'lbs_images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=False)

        basic_info = load_pickle(os.path.join(take_dir, 'basic_info.pkl'))

        scan_frames, scan_rotation = basic_info['scan_frames'], basic_info['rotation']

        Rs, Ts, Ks = d4dress_cameras_to_pytorch3d_cameras(camera_params, camera_ids)

        # build pytorch3d cameras
        R = torch.tensor(Rs, device=self.device, dtype=torch.float32).view(-1, 3, 3)
        T = torch.tensor(Ts, device=self.device, dtype=torch.float32).view(-1, 3)
        cam_K = torch.tensor(Ks, device=self.device, dtype=torch.float32).view(-1, 4, 4)

        cameras = PerspectiveCameras(
            R=R, T=T, K=cam_K,
            image_size=[(1280, 940)],
            device=self.device,
            in_ndc=False
        )
        self.renderer._set_cameras(cameras)


        for frame in scan_frames:
            w_maps = self.solve_frame(take_dir, frame, basic_info)
            # w_maps = w_maps.cpu().numpy()

            # w_sparse = scipy.sparse.csr_matrix(w_maps.reshape(-1, 55))

            w_sparse = w_maps.cpu().to_sparse()

            w_maps = w_maps.cpu().numpy()

            for idx, camera in enumerate(camera_ids):
                save_path = os.path.join(take_dir, 'Capture', camera, 'lbs_images', f'lbs_image-f{frame}.pt')
                # np.save(save_path, w_maps[idx])
                torch.save(w_sparse, save_path)
                # if idx==0:
                    # print(f'saved {save_path}')

        # del w_maps, renderer_output, pytorch3d_mesh
        # del smpl_vertices, smpl_faces, scan_mesh_verts, scan_mesh_faces
        # torch.cuda.empty_cache()  # optional but useful if memory keeps climbing
            # print(f'Saved frame {frame}')

            # w_plot = np.argmax(w_maps, axis=-1)
            # fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            # for i in range(4):
            #     row = i // 2
            #     col = i % 2
            #     axes[row, col].imshow(w_plot[i])
            #     axes[row, col].set_title(f'Camera {i}')
            #     axes[row, col].axis('off')
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig(f'w_map_frame_{frame}.png')
            # import ipdb; ipdb.set_trace()



    def solve_frame(self, take_dir, frame, basic_info):
        # ------------- load SMPL data -------------
        gender = basic_info['gender']
        if gender == 'male':
            smpl_model = self.smpl_male
            lbs_weights = self.smpl_male.lbs_weights
        else:
            smpl_model = self.smpl_female
            lbs_weights = self.smpl_female.lbs_weights

        smpl_ply_fname = os.path.join(take_dir, 'SMPLX', f'mesh-f{frame}_smplx.ply')
        smpl_data = trimesh.load(smpl_ply_fname)
        smpl_vertices, smpl_faces = smpl_data.vertices, smpl_data.faces
        smpl_vertices = torch.tensor(smpl_vertices, device=self.device, dtype=torch.float32)[None]
        smpl_faces = torch.tensor(smpl_faces, device=self.device, dtype=torch.long)[None]



        # ------------- load scan mesh -------------
        scan_mesh_fname = os.path.join(take_dir, 'Meshes_pkl', f'mesh-f{frame}.pkl')
        scan_mesh = load_pickle(scan_mesh_fname)
        scan_mesh_verts = scan_mesh['vertices'][None]
        scan_mesh_faces = scan_mesh['faces'][None]
        
        scan_mesh_verts = torch.tensor(scan_mesh_verts, device=self.device, dtype=torch.float32)
        scan_mesh_faces = torch.tensor(scan_mesh_faces, device=self.device, dtype=torch.long)

        _, idx = self.knn_ptcld(
            Pointclouds(points=scan_mesh_verts), 
            smpl_vertices, 
            K=1
        )
        smpl_weights_flat = lbs_weights.view(-1, 10475, 55)
        idx_expanded = idx.repeat(1, 1, 55)
        scan_w_tensor = torch.gather(smpl_weights_flat, dim=1, index=idx_expanded)


        pytorch3d_mesh = Meshes(
            verts=scan_mesh_verts.repeat(4, 1, 1),
            faces=scan_mesh_faces.repeat(4, 1, 1),
            textures=TexturesVertex(verts_features=scan_w_tensor.repeat(4, 1, 1))
        )
        renderer_output = self.renderer(pytorch3d_mesh)
        w_maps = renderer_output['maps']

        return w_maps 


    def knn_ptcld(self, x, y, K=1):
        with torch.autocast(enabled=False, device_type='cuda'):
            x_lengths, x_normals = None, None
            y_lengths, y_normals = None, None

            x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
            y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
            
            x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=2, K=K)
            dist, idx, _ = x_nn
            return dist, idx
        


if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    solver = Solver()

    take_dir = '/scratch/u5au/chexuan.u5au/4DDress/00122/Inner/Take4'
    # solver.solve_take(id='00122', take_dir=take_dir)
    # solver.solve_id('00122')
    solver.solve_all()
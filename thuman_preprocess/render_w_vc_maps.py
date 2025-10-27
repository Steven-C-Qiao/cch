import os 
import torch
import smplx 
import numpy as np 
from scipy.sparse import csr_matrix, save_npz

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import TexturesVertex
from pytorch3d.transforms import matrix_to_euler_angles, axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points

from core.utils.feature_renderer import FeatureRenderer
from core.data.d4dress_utils import load_pickle
from core.utils.camera_utils import uv_to_pixel_space, custom_opengl2pytorch3d

PATH = '/scratch/u5aa/chexuan.u5aa/datasets/THuman'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

camera_ids = [
    '000', '010', '020', '030', '040', '050', '060', '070', '080', 
    '090', '100', '110', '120', '130', '140', '150', '160', '170', 
    '180', '190', '200', '210', '220', '230', '240', '250', '260', 
    '270', '280', '290', '300', '310', '320', '330', '340', '350', 
]




def knn_ptcld(x, y, K=1):
    with torch.autocast(enabled=False, device_type='cuda'):
        x_lengths, x_normals = None, None
        y_lengths, y_normals = None, None

        x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
        y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
        
        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=2, K=K)
        dist, idx, _ = x_nn
        return dist, idx

def save_w_maps_sparse(w_maps, scan_id, save_path, camera_ids):
    """
    Save w_maps as 36 separate sparse matrices in a subdirectory.
    
    Args:
        w_maps: Tensor of shape (36, 512, 512, 55) - 36 camera views
        scan_id: String identifier for the scan
        save_path: Base path where to save the sparse matrices
        camera_ids: List of camera ID strings to use as filenames
    """
    # Create subdirectory for sparse matrices
    sparse_dir = os.path.join(save_path, f'{scan_id}_w_maps_sparse')
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Convert to numpy and process each camera view
    w_maps_np = w_maps.detach().cpu().numpy()
    
    for camera_idx, camera_id in enumerate(camera_ids):
        # Extract the data for this camera view: (512, 512, 55)
        camera_data = w_maps_np[camera_idx]
        
        # Reshape to (512*512, 55) to treat each pixel as a row
        reshaped_data = camera_data.reshape(-1, 55)
        
        # Convert to sparse matrix (CSR format for efficiency)
        sparse_matrix = csr_matrix(reshaped_data)
        
        # Save as .npz file using camera_id as filename
        camera_filename = os.path.join(sparse_dir, f'{camera_id}.npz')
        save_npz(camera_filename, sparse_matrix)
        
        # print(f"Saved camera {camera_id} sparse matrix: {camera_filename}")
    
    print(f"All 36 sparse matrices saved in: {sparse_dir}")
    return sparse_dir

def render_single(PATH, scan_id, renderer, smpl_model):
    with torch.no_grad():

        smplx_fname = os.path.join(PATH, 'smplx', scan_id, f'smplx_param.pkl')
        smplx_param = np.load(smplx_fname, allow_pickle=True)
        smplx_param['left_hand_pose'] = smplx_param['left_hand_pose'][:, :12]
        smplx_param['right_hand_pose'] = smplx_param['right_hand_pose'][:, :12]
        if scan_id >= '0526':
            global_orient = torch.tensor(smplx_param['global_orient'])
            global_orient = matrix_to_euler_angles(axis_angle_to_matrix(global_orient), 'XYZ') + torch.tensor([-torch.pi/2, 0., 0.])
            smplx_param['global_orient'] = matrix_to_axis_angle(euler_angles_to_matrix(global_orient, 'XYZ')).numpy()

        for k, v in smplx_param.items():
            smplx_param[k] = torch.tensor(v).float().to(device)
        smplx_param.pop('transl')

        smplx_T_param = {
            k: torch.zeros_like(smplx_param[k]) for k in smplx_param.keys()
        }
        smplx_T_param['betas'] = smplx_param['betas']


        smpl_output = smpl_model(**smplx_param, return_full_pose=True)
        smpl_T_output = smpl_model(**smplx_T_param, return_full_pose=True)

        smpl_T_height = smpl_T_output.vertices[:, :, 1].max(dim=-1).values - smpl_T_output.vertices[:, :, 1].min(dim=-1).values

        smpl_T_verts = smpl_T_output.vertices / smpl_T_height[:, None, None] * 1.7



        scan_fname = os.path.join(PATH, 'cleaned', f'{scan_id}.pkl')
        scan = load_pickle(scan_fname)

        scan_verts = torch.tensor(scan['scan_verts'], device=device).float()
        scan_faces = torch.tensor(scan['scan_faces'], device=device).long()

        cam_Rs, cam_Ts = [], []
        for camera_id in camera_ids:

            calib_fname = os.path.join(PATH, 'render_persp/thuman2_36views', scan_id, 'calib', f'{camera_id}.txt')
            calib = np.loadtxt(calib_fname)
            extrinsic = calib[:4].reshape(4,4)
            intrinsic = calib[4:].reshape(4,4)

            intrinsic = uv_to_pixel_space(intrinsic)
            intrinsic[0, 2] = 256
            intrinsic[1, 2] = 256

            extrinsic = torch.tensor(extrinsic).float()
            intrinsic = torch.tensor(intrinsic).float()

            cam_R, cam_T = custom_opengl2pytorch3d(extrinsic)

            cam_Rs.append(cam_R)
            cam_Ts.append(cam_T)

        cam_Rs = torch.stack(cam_Rs)
        cam_Ts = torch.stack(cam_Ts)
        
        cameras = PerspectiveCameras(
            R=cam_Rs,
            T=cam_Ts, 
            # K=cam_K.flatten(0, 1),
            focal_length=724.0773,
            principal_point=[(256, 256),],
            image_size=[(512, 512),],
            device=device,
            in_ndc=False
        )
        renderer._set_cameras(cameras)


        # dists, idx = knn_ptcld(
        #     Pointclouds(points=scan_verts[None]), 
        #     smpl_output.vertices,
        #     K=1
        # )
        # smpl_weights_flat = smpl_model.lbs_weights[None]
        # idx_expanded = idx.repeat(1, 1, 55)
        # scan_w_tensor = torch.gather(smpl_weights_flat, dim=1, index=idx_expanded)
        # # scan_w = [scan_w_tensor[i, :len(verts), :] for i, verts in enumerate(scan_verts)]
        # scan_w = scan_w_tensor

        # # Render skinning weight pointmaps
        # pytorch3d_mesh = Meshes(
        #     verts=scan_verts[None].repeat(len(camera_ids), 1, 1),
        #     faces=scan_faces[None].repeat(len(camera_ids), 1, 1),
        #     textures=TexturesVertex(verts_features=scan_w.repeat(len(camera_ids), 1, 1))
        # )

        # renderer_output = renderer(pytorch3d_mesh)
        # w_maps = renderer_output['maps']


        pytorch3d_smpl_mesh = Meshes(
            verts=smpl_output.vertices.repeat(len(camera_ids), 1, 1),
            faces=torch.tensor(smpl_model.faces, device=device)[None].repeat(len(camera_ids), 1, 1),
            textures=TexturesVertex(verts_features=smpl_T_verts.repeat(len(camera_ids), 1, 1))
        )

        renderer_output = renderer(pytorch3d_smpl_mesh)
        vc_maps = renderer_output['maps']

    return None, vc_maps




if __name__ == "__main__":
    renderer = FeatureRenderer(image_size=(512, 512))

    smpl_model = smplx.create(
        model_path='model_files/',
        model_type='smplx',
        gender='neutral',
        num_pca_comps=12,
        num_betas=10,
    ).to(device)

    scan_ids = os.listdir(os.path.join(PATH, 'model'))

    failed_ids = []
    
    for scan_id in sorted(scan_ids):
        if scan_id >= '1200':
            try:
                print(f"Processing scan_id: {scan_id}")
                w_maps, vc_maps = render_single(PATH, scan_id, renderer, smpl_model)

                save_path = os.path.join(PATH, 'render_persp/thuman2_36views', scan_id)
                
                # Save w_maps as 36 separate sparse matrices using camera_ids as filenames
                # sparse_dir = save_w_maps_sparse(w_maps, scan_id, save_path, camera_ids)
                
                # Still save the original dense format for vc_maps
                np.save(
                    os.path.join(save_path, f'{scan_id}_vc_maps_normalised_170cm'),
                    vc_maps.detach().cpu().numpy()
                )

                # import matplotlib.pyplot as plt
                
                # # Create figure with subplots in a row
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # # Plot w_maps and vc_maps
                # axes[0].imshow(np.argmax(w_maps[0].detach().cpu().numpy(), axis=-1))
                # axes[0].set_title('Skinning Weight Maps')
                # axes[0].axis('off')
                
                # axes[1].imshow(vc_maps[0].detach().cpu().numpy())
                # axes[1].set_title('Vertex Color Maps') 
                # axes[1].axis('off')
                
                # # Save figure
                # plt.savefig(f'{scan_id}_maps.png', bbox_inches='tight')
                # plt.close()
            except:
                failed_ids.append(scan_id)
                print(f"Failed to process scan_id: {scan_id}")
                continue

    if failed_ids:
        print(f"Failed to process the following scan_ids: {failed_ids}")
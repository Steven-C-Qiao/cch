import os 
import trimesh
import numpy as np
import cv2
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import matplotlib.pyplot as plt
from typing import Optional

from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    PointsRasterizationSettings,
    SoftPhongShader,
)

def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

class Solver:
    def __init__(self, device):
        self.device = device
        self.img_size = 1024
        self.PATH_TO_DATASET = "/scratches/kyuban/cq244/datasets/4DDress"


    def filter_mesh(self, mesh, id):
        """
        Filter mesh to obtain only visible portions by rendering from multiple viewpoints
        and determining which faces are visible using renderer fragments.
        
        Args:
            mesh: trimesh.Trimesh object containing the full mesh with multiple submeshes
            id: subject identifier for saving results
            
        Returns:
            filtered_mesh: trimesh.Trimesh object containing only visible portions
        """
        # Convert trimesh to pytorch3d mesh for rendering
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64).to(self.device)
        pytorch3d_mesh = Meshes(verts=[vertices], faces=[faces]).to(self.device)
        
        # Initialize set to store visible faces
        visible_faces = set()
        
        # Setup rasterizer for rendering
        rasterizer = MeshRasterizer(
            cameras=PerspectiveCameras(),
            raster_settings=RasterizationSettings(
                image_size=self.img_size, 
                max_faces_per_bin=1000000, 
                bin_size=None, 
                blur_radius=0.
            )
        ).to(self.device)

        # Generate multiple camera viewpoints around the mesh
        mesh_center = mesh.centroid
        mesh_size = np.max(mesh.bounds[1] - mesh.bounds[0])
        camera_distance = mesh_size * 1.5
        
        # Create cameras at different azimuth angles and elevations
        azim_angles = np.linspace(0, 360, 12)  # 12 horizontal viewpoints
        elev_angles = [-30, 0, 30]  # 3 elevation levels
        
        print(f"Rendering from {len(azim_angles) * len(elev_angles)} camera viewpoints...")
        
        camera_count = 0
        total_cameras = len(azim_angles) * len(elev_angles)
        
        for elev in elev_angles:
            for azim in azim_angles:
                camera_count += 1
                print(f"Processing camera {camera_count}/{total_cameras} (elev={elev}, azim={azim})")
                
                # Generate camera position
                R, T = look_at_view_transform(
                    dist=camera_distance,
                    elev=elev,
                    azim=azim
                )
                
                # Setup camera intrinsics
                K = torch.tensor([[[self.img_size/2., 0., self.img_size/2., 0],
                                  [0., self.img_size/2., self.img_size/2., 0],
                                  [0., 0., 0., 1.],
                                  [0., 0., 1., 0.]]]).to(self.device)
                
                # Create camera
                camera = PerspectiveCameras(
                    R=R,
                    T=T,
                    K=K,
                    in_ndc=False,
                    image_size=[(self.img_size, self.img_size)]
                ).to(self.device)
                
                try:
                    # Render the mesh and get fragments
                    fragments = rasterizer(pytorch3d_mesh, cameras=camera)
                    
                    # Get face indices that are visible (not background)
                    face_idx = fragments.pix_to_face[..., 0]  # Get the closest face for each pixel
                    valid_mask = face_idx != -1  # -1 indicates background/no face
                    
                    # Get unique face indices that are visible
                    visible_face_indices = face_idx[valid_mask].unique()
                    
                    # Add to our set of visible faces
                    for face_idx in visible_face_indices:
                        if face_idx.item() != -1:  # Skip background
                            visible_faces.add(face_idx.item())
                                
                except Exception as e:
                    print(f"Warning: Failed to process camera {camera_count}: {e}")
                    continue
        
        print(f"Found {len(visible_faces)} visible faces out of {len(mesh.faces)} total")
        
        # Convert visible faces set to sorted list
        visible_faces_list = sorted(list(visible_faces))
        
        # Create filtered mesh with only visible faces
        filtered_vertices, filtered_faces = mesh.vertices, mesh.faces[visible_faces_list]
        
        # Remove unreferenced vertices
        filtered_mesh = trimesh.Trimesh(vertices=filtered_vertices, faces=filtered_faces)
        filtered_mesh.remove_unreferenced_vertices()
        
        # Save the visible face indices for reference
        # save_path = os.path.join(self.PATH_TO_DATASET, '_4D-DRESS_Template', id, f'{id}_outer_faces.npy')
        # np.save(save_path, np.array(visible_faces_list))
        
        # Also save the filtered mesh
        mesh_save_path = os.path.join(self.PATH_TO_DATASET, '_4D-DRESS_Template', id, f'filtered_w_outer.ply')
        filtered_mesh.export(mesh_save_path)
        
        print(f"Filtered mesh saved with {len(filtered_mesh.vertices)} vertices and {len(filtered_mesh.faces)} faces")
        
        return filtered_mesh
            


    def render_depth_map(self, mesh, rasterizer, camera=None):
        if camera is not None:
            rasterizer.cameras = camera

        fragments = rasterizer(mesh)
        depth_map = fragments.zbuf[..., 0]
        return depth_map




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_TO_DATASET = "/scratches/kyuban/cq244/datasets/4DDress"

    ids = [ '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137', 
            '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156', 
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175', 
            '00176', '00179', '00180', '00185', '00187', '00188', '00190', '00191']

    for id in ids:
        print(f'Processing {id}')
    
        # ---- template mesh ----
        template_dir = os.path.join(PATH_TO_DATASET, '_4D-DRESS_Template', id)
            
        upper_mesh = trimesh.load(os.path.join(template_dir, 'upper.ply'))
        body_mesh = trimesh.load(os.path.join(template_dir, 'body.ply'))
        outer_mesh = trimesh.load(os.path.join(template_dir, 'outer.ply'))
        if os.path.exists(os.path.join(template_dir, 'lower.ply')):
            lower_mesh = trimesh.load(os.path.join(template_dir, 'lower.ply'))
            full_mesh = trimesh.util.concatenate([body_mesh, lower_mesh, upper_mesh, outer_mesh])
            clothing_mesh = trimesh.util.concatenate([lower_mesh, upper_mesh, outer_mesh])

        else:
            full_mesh = trimesh.util.concatenate([body_mesh, upper_mesh, outer_mesh])
            clothing_mesh = trimesh.util.concatenate([upper_mesh, outer_mesh])

        solver = Solver(device=device)
        solver.filter_mesh(full_mesh, id)
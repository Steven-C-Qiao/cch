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
    def __init__(self, body_mesh, clothing_mesh, device):
        self.body_mesh = body_mesh
        self.clothing_mesh = clothing_mesh
        self.device = device
        self.img_size = 1024


    def filter_visible_vertices(self, id):
        visible_vertices = set() 

        rasterizer=MeshRasterizer(
            cameras=PerspectiveCameras(),
            raster_settings=RasterizationSettings(image_size=self.img_size, max_faces_per_bin=1000000, bin_size=None, blur_radius=0.)
        ).to(self.device)

        # Get R, T matrices for all camera positions looking at mesh center
        R, T = look_at_view_transform(dist = 1.5,
                                      azim = np.linspace(0, 360, 10))
        
        K = torch.tensor([[[self.img_size/2., 0., self.img_size/2., 0],
                          [0., self.img_size/2., self.img_size/2., 0],
                          [0., 0., 0., 1.],
                          [0., 0., 1., 0.]]]).to(self.device)
        K2 = torch.tensor([[[self.img_size/2., 0., self.img_size/2., 0],
                          [0., -self.img_size/2., self.img_size/2., 0],
                          [0., 0., 0., 1.],
                          [0., 0., 1., 0.]]]).to(self.device)

        cameras = PerspectiveCameras(
            R=R,
            T=T,
            K=K,
            in_ndc=False,
            image_size=[(self.img_size, self.img_size)]
        ).to(self.device)


        for i, camera in enumerate(cameras):
            body_depth = self.render_depth_map(self.body_mesh, rasterizer, camera).squeeze()
            clothing_depth = self.render_depth_map(self.clothing_mesh, rasterizer, camera).squeeze()
            clothing_depth[clothing_depth==-1] = 10
            # import ipdb; ipdb.set_trace()
            # clothing_depth[clothing_depth==10] = torch.max(clothing_depth)

            cameras = PerspectiveCameras(
                R=R.transpose(-1, -2),
                T=T,
                K=K2,
                in_ndc=False,
                image_size=[(self.img_size, self.img_size)]
            ).to(self.device)
            camera = cameras[i]

            body_screen_coords = camera.transform_points(points=self.body_mesh.verts_packed())
            body_screen_coords = body_screen_coords.cpu().detach().numpy()


            # Visualize visible vertices in 3D
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(8, 8))

            # # Plot all vertices in red
            # all_verts = body_screen_coords
            # plt.scatter(all_verts[:, 0], all_verts[:, 1], s=1, c='red', alpha=0.3, label='All vertices')

            # plt.legend()
            # plt.gca().set_aspect('equal')
            
            # plt.savefig(f'screen_coords_{i}.png')
            # plt.close()




            # Check each body vertex
            for j, (vertex, screen_coord) in enumerate(zip(self.body_mesh.verts_packed(), body_screen_coords)):
                x, y = int(screen_coord[0]), int(screen_coord[1])
                x_h, y_h = x + 1, y + 1
                x_l, y_l = x - 1, y - 1
                if 0 <= x < self.img_size and 0 <= y < self.img_size and 0 <= x_h < self.img_size and 0 <= y_h < self.img_size and 0 <= x_l < self.img_size and 0 <= y_l < self.img_size:
                    # If body is visible (not occluded by clothing) at this pixel
                    if body_depth[y, x] < clothing_depth[y, x] - 0.001 and body_depth[y_h, x_h] < clothing_depth[y_h, x_h] - 0.001 and body_depth[y_l, x_l] < clothing_depth[y_l, x_l] - 0.001  :  # Small tolerance
                        visible_vertices.add(j)  # Set vertex as visible

            # import matplotlib.pyplot as plt
            # # Visualize depth maps
            # body_depth = body_depth.squeeze().cpu().detach().numpy()
            # clothing_depth = clothing_depth.squeeze().cpu().detach().numpy()


            
            # plt.figure(figsize=(12, 6))
            
            # plt.subplot(121)
            # plt.imshow(body_depth)
            # plt.colorbar(label='Depth')
            # plt.title('Body Depth Map')
            
            # plt.subplot(122)
            # plt.imshow(clothing_depth)
            # plt.colorbar(label='Depth') 
            # plt.title('Clothing Depth Map')
            
            # plt.tight_layout()
            # plt.savefig(f'depth_maps_camera_{i+1}.png')
            # plt.close()

        print(f"Found {len(visible_vertices)} visible vertices out of {len(self.body_mesh.verts_packed())} total")

        # Visualize visible vertices in 3D
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot all vertices in red
        all_verts = self.body_mesh.verts_packed().cpu().detach().numpy()
        # ax.scatter(all_verts[:, 0], all_verts[:, 1], all_verts[:, 2], s=1, c='red', alpha=0.3, label='All vertices')
        
        # Plot visible vertices in blue
        visible_verts = all_verts[list(visible_vertices)]
        ax.scatter(visible_verts[:, 0], visible_verts[:, 1], visible_verts[:, 2], s=1, c='blue', alpha=0.8, label='Visible vertices')
        

        clothes_verts = self.clothing_mesh.verts_packed().cpu().detach().numpy()
        ax.scatter(clothes_verts[:, 0], clothes_verts[:, 1], clothes_verts[:, 2], s=0.5, c='green', alpha=0.8, label='Clothes vertices')

        ax.view_init(elev=10, azim=20, vertical_axis='y')
        ax.legend()
        ax.set_title('Body Mesh Vertices (Red: All, Blue: Visible)')
        ax.set_box_aspect([1,1,1])
        ax.set_aspect('equal')
        # plt.show()
        # plt.close()
        plt.savefig('visible_vertices.png')
        plt.close()

        # Convert visible vertices set to sorted numpy array and save
        visible_vertices_array = np.array(sorted(list(visible_vertices)))
        save_path = os.path.join('/scratches/kyuban/cq244/CCH/cch/model_files/4DDress_visible_vertices', f'{id}_outer.npy')
        np.save(save_path, visible_vertices_array)


        # return list(visible_vertices)
            


    def render_depth_map(self, mesh, rasterizer, camera=None):
        if camera is not None:
            rasterizer.cameras = camera

        fragments = rasterizer(mesh)
        depth_map = fragments.zbuf[..., 0]
        return depth_map


# Example usage function
def main(body_mesh, clothing_mesh, id):
    """
    Example of how to use the ray casting vertex filtering.
    """

    # Filter visible vertices
    solver = Solver(body_mesh.to(device), clothing_mesh.to(device), device)
    solver.filter_visible_vertices(id)
    
    # Create new mesh with only visible vertices
    # visible_body_mesh = body_mesh.copy()
    # visible_body_mesh.update_vertices(visible_indices)
    # visible_body_mesh.remove_unreferenced_vertices()
    
    # print(f"Visible body mesh: {len(visible_body_mesh.vertices)} vertices")
    
    # # Save result
    # visible_body_mesh.export('visible_body.obj')
    # print("Saved visible body mesh to 'visible_body.obj'")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_TO_DATASET = "/scratches/kyuban/cq244/datasets/4DDress"

    ids = [ '00122', '00123', '00127', '00129', '00134', '00135', '00136', '00137', 
            '00140', '00147', '00148', '00149', '00151', '00152', '00154', '00156', 
            '00160', '00163', '00167', '00168', '00169', '00170', '00174', '00175', 
            '00176', '00179', '00180', '00185', '00187', '00188', '00190', '00191']

    outer = True 
    
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

        # Convert trimeshes to pytorch3d meshes
        body_verts = torch.tensor(body_mesh.vertices, dtype=torch.float32)
        body_faces = torch.tensor(body_mesh.faces, dtype=torch.int64)
        body_mesh_p3d = Meshes(
            verts=[body_verts],
            faces=[body_faces],
        )

        clothing_verts = torch.tensor(clothing_mesh.vertices, dtype=torch.float32) 
        clothing_faces = torch.tensor(clothing_mesh.faces, dtype=torch.int64)
        clothing_mesh_p3d = Meshes(
            verts=[clothing_verts],
            faces=[clothing_faces],
        )

        main(body_mesh_p3d, clothing_mesh_p3d, id=id)
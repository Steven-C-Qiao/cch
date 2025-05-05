import torch
import torch.nn as nn
import pytorch3d
import smplx 
import pytorch_lightning as pl
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
import numpy as np
import matplotlib.pyplot as plt


CAPE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000001.npz'



class SurfaceNormalRenderer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.renderer = self.create_normal_renderer()

    def create_normal_renderer(self):
        # Create a renderer for surface normals
        R, T = look_at_view_transform(dist=2., elev=-10, azim=0, degrees=True)
        # R, T = look_at_view_transform(at=torch.tensor([0, 0, 2.]), degrees=True)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
        
        # Create rasterization settings
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        # Create a renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                cameras=cameras,
            )
        )
        
        return renderer

    def forward(self, vertices, faces, R=None, T=None):
        """
        Render surface normals 

        Args:
            vertices: (B, V, 3)
            faces: (F, 3)
            R: (B, N, 3, 3); N views
            T: (B, N, 3)
        Returns:
            normals: (B, N, 3, H, W)
        """
        B = vertices.shape[0]
        N = R.shape[1] if R is not None else 1

        vertices = vertices.repeat_interleave(N, dim=0)
        faces = faces[None].repeat_interleave(B * N, dim=0)

        R = R.view(B * N, 3, 3)
        T = T.view(B * N, 3)

        mesh = Meshes(
            verts=vertices,
            faces=faces,
        )
        surface_normals = mesh.verts_normals_padded()

        if R is not None and T is not None:
            cameras = FoVPerspectiveCameras(R=R, T=T).to(R)
            self.renderer.rasterizer.cameras = cameras
            self.renderer.shader.cameras = cameras

            print(R.shape)
            print(surface_normals.shape)
            
            surface_normals = (torch.bmm(R, surface_normals.permute(0, 2, 1)).permute(0, 2, 1))
            
            # invert z axis as per opencv? convention
            surface_normals[:, 2] *= -1


        # normalise to 0 1
        surface_normals = (surface_normals + 1) / 2
        mesh = Meshes(
            verts=vertices,
            faces=faces,
            textures=TexturesVertex(verts_features=surface_normals)
        )

        import ipdb; ipdb.set_trace()

        images = self.renderer(mesh)
        
        # Extract normals from the rendered image
        # The RGB channels correspond to the XYZ components of the normal
        
        import ipdb; ipdb.set_trace()
        normals = images[0, ..., :3].cpu().numpy()
        
        # Normalize the normals
        normals = normals / (np.linalg.norm(normals, axis=2, keepdims=True) + 1e-6)
        
        return normals

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    smpl_model = smplx.create(
        model_path='model_files',
        model_type='smpl',
        gender='neutral',
        device=device
    )
    smpl_faces = smpl_model.faces


    renderer = SurfaceNormalRenderer(device=device)


    data = np.load(CAPE_PATH)

    for key, value in data.items():
        print(key, value.shape)

    print(data['transl'])

    vertices = torch.tensor(data['v_posed'], dtype=torch.float32)
    faces = torch.tensor(smpl_faces, dtype=torch.int32)


    at = np.array(data['transl'])[None]
    at[:, 1] -= 0.2
    num_angles = 6

    fig, axs = plt.subplots(1, num_angles, figsize = (num_angles * 4, 4))
    for i, angle in enumerate(np.linspace(0, 360, num_angles)):
        R, T = look_at_view_transform(dist=2., elev=0, at=at, azim=angle, degrees=True)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
            
        normals = renderer.forward(vertices, faces, cameras)
        axs[i].imshow(normals)

    # Save before showing
    plt.savefig('tinkering/surface_normals_obj_coord.png')
    plt.show()
    plt.close()

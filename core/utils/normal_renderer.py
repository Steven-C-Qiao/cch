import torch
import numpy as np
import torch.nn as nn
import smplx 
import pytorch_lightning as pl
from einops import rearrange

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams

smpl_model = smplx.create(
    model_path='model_files',
    model_type='smpl',
    gender='neutral',
)
smpl_faces = torch.tensor(smpl_model.faces, dtype=torch.int32)



class SimpleShader(nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image
    

class SurfaceNormalRenderer(pl.LightningModule):
    def __init__(self, image_size=(256, 192)):
        super().__init__()
        self.image_size = image_size

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
            shader=SimpleShader()
        )

        self.register_buffer('faces', smpl_faces)
    

    def forward(self, vertices, R=None, T=None, faces=None):
        """
        Args:
            vertices: (B, N, V, 3)
            faces: (F, 3)
            R: (B, N, 3, 3); N views
            T: (B, N, 3)
        Returns:
            normals: (B, N, 3, H, W)
        """
        B, N, _, _ = vertices.shape

        if faces is None:
            faces = self.faces

        vertices = vertices.view(B * N, -1, 3)
        faces = faces[None].repeat(B * N, 1, 1)

        R = R.view(B * N, 3, 3)
        T = T.view(B * N, 3)

        mesh = Meshes(
            verts=vertices,
            faces=faces,
        )
        surface_normals = mesh.verts_normals_padded()

        if R is not None and T is not None:
            cameras = FoVPerspectiveCameras(R=R, T=T).to(self.device)
            self.renderer.rasterizer.cameras = cameras
            self.renderer.shader.cameras = cameras

            surface_normals = torch.bmm(R.permute(0, 2, 1), surface_normals.permute(0, 2, 1)).permute(0, 2, 1)

            # invert z axis as per opencv? convention
            surface_normals[..., 2] *= -1


        # normalise to 0 1
        surface_normals = (surface_normals + 1) / 2
        mesh = Meshes(
            verts=vertices,
            faces=faces,
            textures=TexturesVertex(verts_features=surface_normals)
        )

        images = self.renderer(mesh)
        
        # discard alpha channel
        normals = images[..., :3].cpu().numpy()

        return rearrange(normals, '(b n) c h w -> b n c h w', b=B, n=N)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sample_utils import sample_cameras

    CAPE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000001.npz'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_angles = 6


    data = np.load(CAPE_PATH)

    vertices = torch.tensor(data['v_posed'], dtype=torch.float32)

    at = torch.tensor(data['transl']).float()[None]
    at[:, 1] -= 0.2    
    R, T = sample_cameras(1, num_angles, at)


    renderer = SurfaceNormalRenderer(image_size=(256, 192))

    normals = renderer(vertices[None], 
                               R=R, 
                               T=T)

    n_subplots = normals.shape[1]
    fig, axs = plt.subplots(1, n_subplots, figsize = (n_subplots * 4, 4))
    for i in range(n_subplots):
        axs[i].imshow(normals[0, i])
    plt.savefig('tinkering/test_normal_renderer.png')
    plt.show()
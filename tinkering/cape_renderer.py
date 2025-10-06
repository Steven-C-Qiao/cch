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
    TexturesVertex,
    SoftPhongShader,
    PointLights
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams

class SimpleRenderer(nn.Module):
    def __init__(self, image_size=(256, 192), device="cpu"):
        super().__init__()
        self.image_size = image_size
        self.device = device
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Create a point light source at (0, 0, 1)
        lights = PointLights(
            device=device,
            location=[[0.0, 0.0, 10.0]],
            ambient_color=[[0.5, 0.5, 0.5]],  # Ambient light color
            diffuse_color=[[0.7, 0.7, 0.7]],  # Diffuse light color
            specular_color=[[0.3, 0.3, 0.3]]  # Specular light color
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                # lights=lights,
            )
        ).to(self.device)

    def forward(self, vertices, R=None, T=None, textures=None):
        if textures is None:
            # Create light blue textures by setting RGB values
            light_blue = torch.ones_like(vertices) * torch.tensor([0.6, 0.8, 1.0]).to(self.device)
            textures = TexturesVertex(verts_features=light_blue).to(self.device)

        # set cameras
        cameras = FoVPerspectiveCameras(
            R=R,
            T=T,
            device=self.device
        )
        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras

        mesh = Meshes(
            verts=vertices,
            faces=smpl_faces[None].repeat(vertices.shape[0], 1, 1),
            textures=textures
        )

        return self.renderer(mesh)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    from pytorch3d.renderer.cameras import look_at_view_transform

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CAPE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/'


    smpl_model = smplx.create(
        model_path='model_files',
        model_type='smpl',
        gender='neutral',
    )
    smpl_faces = torch.tensor(smpl_model.faces, dtype=torch.int32).to(device)




    num_views = 4
    dist=torch.ones(num_views) * 2.
    elev=torch.ones(num_views) * 0.0
    at = torch.tensor([0, -0.4, 0])[None].repeat(num_views, 1).to(device)
    azim = torch.linspace(0, 360, num_views)
    R, T = look_at_view_transform(  dist=dist, 
                                    elev=elev, 
                                    azim=azim, 
                                    at=at,
                                    degrees=True)
    
    all_verts = []
    for i in [1, 50, 100, 150]:
        # vertices = np.load(CAPE_PATH + f'longshort_ATUsquat.{i:06d}.npz')['v_posed']
        vertices = np.load(CAPE_PATH + f'longshort_ATUsquat.000001.npz')['v_cano']
        vertices = torch.tensor(vertices, dtype=torch.float32).to(device)
        all_verts.append(vertices)
    all_verts = torch.stack(all_verts, dim=0)

    renderer = SimpleRenderer(image_size=(512, 512), device=device)
    ret = renderer(all_verts, R=R, T=T)
    
    
    fig = plt.figure(figsize=(12, 5))
    for i in range(num_views):
        ax = fig.add_subplot(1, 4, i+1)
        ax.imshow(ret.cpu().numpy()[i, ..., :-1])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('tinkering/simple_renderer.png')
    plt.show()


    # num_views = 250
    # dist=torch.ones(num_views) * 2.5
    # elev=torch.ones(num_views) * 0.0
    # at = torch.tensor([0, 0.4, 0])[None].repeat(num_views, 1).to(device)
    # azim = torch.linspace(0, 360, num_views)
    # R, T = look_at_view_transform(  dist=dist, 
    #                                 elev=elev, 
    #                                 azim=azim, 
    #                                 at=at,
    #                                 degrees=True)

    # all_vertices = []
    # for i in range(1, num_views+1):
    #     vertices = np.load(CAPE_PATH + f'longshort_ATUsquat.{i:06d}.npz')['v_posed']
    #     vertices = torch.tensor(vertices, dtype=torch.float32).to(device)
    #     all_vertices.append(vertices)

    # all_vertices = torch.stack(all_vertices, dim=0)

    # renderer = SimpleRenderer(image_size=(512, 512), device=device)
    # ret = renderer( all_vertices, 
    #                 R=R.to(device), 
    #                 T=T.to(device))
    # imgs_for_gif = (ret.cpu().numpy()[..., :-1] * 255).astype(np.uint8)

    # import imageio
    # imageio.mimsave('tinkering/test_renderer.gif', imgs_for_gif, fps=30)

    # import ipdb; ipdb.set_trace()
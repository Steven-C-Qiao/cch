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
        
        # If we have more than 3 channels, return all channels
        if texels.shape[-1] > 3:
            # Use the alpha channel from fragments for masking
            alpha = fragments.zbuf[..., 0] > -1

            # Expand alpha to match number of channels
            alpha = alpha.unsqueeze(-1).unsqueeze(-1) # no need to expand as texels as same alpha for all channels

            # Return all channels with alpha mask
            return torch.cat([texels, alpha], dim=-1).squeeze() # (N, H, W, C+1)
        
        # For RGB textures, use standard blending
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image for RGB, or (N, H, W, C+1) for multi-channel
    

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
    

    def forward(self, vertices, R=None, T=None, faces=None, skinning_weights=None):
        """
        Args:
            vertices: (B, N, V, 3)
            faces: (F, 3)
            R: (B, N, 3, 3); N views
            T: (B, N, 3)
            skinning_weights: (B, N, V, 24)
        Returns:
            normals: (B, N, 3, H, W)
            mask: (B, N, 1, H, W)
        """
        B, N, _, _ = vertices.shape
        ret = {}

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
        
        # Get normal maps and alpha channel
        normals = images[..., :3].cpu().numpy()
        mask = images[..., 3:4].cpu().numpy()  # Extract alpha channel as mask

        ret['normals'] = rearrange(normals, '(b n) h w c -> b n h w c', b=B, n=N)
        ret['masks'] = rearrange(mask, '(b n) h w c -> b n h w c', b=B, n=N)


        colors = (vertices - vertices.min()) / (vertices.max() - vertices.min())

        mesh.textures = TexturesVertex(verts_features=colors) # render a color map 
        images = self.renderer(mesh)
        color_map = images[..., :3].cpu().numpy()
        ret['color_maps'] = rearrange(color_map, '(b n) h w c -> b n h w c', b=B, n=N)


        # render the skinning weight maps 
        if skinning_weights is not None:
            skinning_weights = rearrange(skinning_weights, 'b n v k -> (b n) v k')
            mesh.textures = TexturesVertex(verts_features=skinning_weights)
            images = self.renderer(mesh)
            skinning_weights_map = images[..., :-1].cpu().numpy()
            ret['skinning_weights_maps'] = rearrange(skinning_weights_map, '(b n) h w c -> b n h w c', b=B, n=N)

        return ret

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sample_utils import sample_cameras

    CAPE_PATH_0 = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000001.npz'
    CAPE_PATH_1 = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000050.npz'
    CAPE_PATH_2 = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000100.npz'
    CAPE_PATH_3 = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000150.npz'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_angles = 4


    data_0 = np.load(CAPE_PATH_0)
    data_1 = np.load(CAPE_PATH_1)
    data_2 = np.load(CAPE_PATH_2)
    data_3 = np.load(CAPE_PATH_3)

    smpl_model = smplx.create(
        model_path='model_files',
        model_type='smpl',
        gender='neutral',
    )
    skinning_weights = smpl_model.lbs_weights[None, None].repeat(1, 4, 1, 1)

    v_posed = np.stack([data_0['v_posed'], data_1['v_posed'], data_2['v_posed'], data_3['v_posed']], axis=0)[None]

    vertices = torch.tensor(v_posed, dtype=torch.float32)
    print(vertices.shape)

    at = torch.tensor(np.stack([data_0['transl'], data_1['transl'], data_2['transl'], data_3['transl']], axis=0)[None]).float()
    at[:, -1] -= 0.2    
    R, T = sample_cameras(1, num_angles, at)


    renderer = SurfaceNormalRenderer(image_size=(256, 192))

    ret = renderer(vertices, 
                    R=R, 
                    T=T,
                    skinning_weights=skinning_weights)
    normals = ret['normals']
    mask = ret['masks']
    color_map = ret['color_maps']
    w = ret['skinning_weights_maps'] # B, N, H, W, 24

    w = w.argmax(axis=-1)
    w = (w - w.min()) / (w.max() - w.min())

    

    n_subplots = normals.shape[1]
    fig, axs = plt.subplots(4, n_subplots, figsize=(n_subplots * 4, 12))
    for i in range(n_subplots):
        axs[0, i].imshow(normals[0, i])
        axs[1, i].imshow(mask[0, i], cmap='gray')
        axs[2, i].imshow(color_map[0, i])
        axs[3, i].imshow(w[0, i])
    plt.tight_layout()
    for ax in axs.flatten():
        ax.axis('off')
    plt.savefig('tinkering/test_renderer.png')
    plt.show()
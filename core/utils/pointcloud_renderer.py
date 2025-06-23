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
        """
        pix_to_face: LongTensor of shape (N, image_size, image_size, faces_per_pixel) 
                     giving the indices of the nearest faces at each pixel, sorted in ascending z-order.
        """
        # ------------------- vertex visibility -------------------
        pix_to_face = fragments.pix_to_face
        # (F, 3) where F is the total number of faces across all the meshes in the batch
        packed_faces = meshes.faces_packed() 
        # (V, 3) where V is the total number of verts across all the meshes in the batch
        packed_verts = meshes.verts_packed() 
        vertex_visibility_map = torch.zeros(packed_verts.shape[0])   # (V,)

        # Indices of unique visible faces
        visible_faces = pix_to_face.unique()[1:]   # (num_visible_faces )

        # Get Indices of unique visible verts using the vertex indices in the faces
        visible_verts_idx = packed_faces[visible_faces]    # (num_visible_faces,  3)
        unique_visible_verts_idx = torch.unique(visible_verts_idx)   # (num_visible_verts, )

        # Update visibility indicator to 1 for all visible vertices 
        vertex_visibility_map[unique_visible_verts_idx] = 1.0

        vertex_visibility_map = vertex_visibility_map.view(pix_to_face.shape[0], -1)
        self.vertex_visibility = vertex_visibility_map


        # ------------------- blending -------------------
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
    

    def forward(self, vertices, R=None, T=None, faces=None, skinning_weights=None, first_frame_v_cano=None, part_segmentation=None):
        """
        Args:
            vertices: (B, N, V, 3)
            faces: (F, 3)
            R: (B, N, 3, 3); N views
            T: (B, N, 3)
            skinning_weights: (B, N, V, 24)
            first_frame_v_cano: (B, V, 3)
        Returns:
            normals: (B, N, 3, H, W)
            mask: (B, N, 1, H, W)
        """
        B, N, _, _ = vertices.shape
        ret = {}

        if faces is None:
            faces = self.faces

        if first_frame_v_cano is not None:
            assert len(first_frame_v_cano.shape) == 3
            first_frame_v_cano = first_frame_v_cano.repeat_interleave(N, dim=0)

        vertices = vertices.view(B * N, -1, 3)
        faces = faces[None].repeat(B * N, 1, 1)

        R = R.view(B * N, 3, 3)
        T = T.view(B * N, 3)

        # ------------------- surface normals -------------------
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
        normals = images[..., :3]#.cpu().numpy()
        ret['normals'] = rearrange(normals, '(b n) h w c -> b n h w c', b=B, n=N)

        # Store vertex visibility information
        vertex_visibility = self.renderer.shader.vertex_visibility
        ret['vertex_visibility'] = rearrange(vertex_visibility, '(b n) v -> b n v', b=B, n=N)

        # ------------------- masks -------------------
        mask = images[..., 3:4]#.cpu().numpy()  # Extract alpha channel as mask
        ret['masks'] = rearrange(mask, '(b n) h w c -> b n h w c', b=B, n=N)

        # ------------------- color maps -------------------
        colors = (vertices - vertices.min()) / (vertices.max() - vertices.min())

        mesh.textures = TexturesVertex(verts_features=colors) # render a color map 
        images = self.renderer(mesh)
        color_map = images[..., :3] #.cpu().numpy()
        ret['color_maps'] = rearrange(color_map, '(b n) h w c -> b n h w c', b=B, n=N)


        # ------------------- skinning weight maps -------------------
        if skinning_weights is not None:
            skinning_weights = rearrange(skinning_weights, 'b n v k -> (b n) v k')
            mesh.textures = TexturesVertex(verts_features=skinning_weights)
            images = self.renderer(mesh)
            skinning_weights_map = images[..., :-1]#.cpu().numpy()
            ret['skinning_weights_maps'] = rearrange(skinning_weights_map, '(b n) h w c -> b n h w c', b=B, n=N)

        # ------------------- canonical color maps -------------------
        if first_frame_v_cano is not None:
            mesh.textures = TexturesVertex(verts_features=first_frame_v_cano)
            images = self.renderer(mesh)
            canonical_color_map = images[..., :3]#.cpu().numpy()
            ret['canonical_color_maps'] = rearrange(canonical_color_map, '(b n) h w c -> b n h w c', b=B, n=N)

        if part_segmentation is not None:
            mesh.textures = TexturesVertex(verts_features=part_segmentation)
            images = self.renderer(mesh)
            part_segmentation_map = images[..., :3]#.cpu().numpy()
            ret['part_segmentation_maps'] = rearrange(part_segmentation_map, '(b n) h w c -> b n h w c', b=B, n=N)

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
                    skinning_weights=skinning_weights,
                    first_frame_v_cano=torch.tensor(v_posed[:, 0]).float())
    normals = ret['normals']
    mask = ret['masks']
    color_map = ret['color_maps']
    canonical_color_map = ret['canonical_color_maps']
    vertex_visibility = ret['vertex_visibility'] # B, N, V
    
    w = ret['skinning_weights_maps'] # B, N, H, W, 24
    w = w.argmax(axis=-1)
    w = (w - w.min()) / (w.max() - w.min())

    

    n_subplots = normals.shape[1]
    fig, axs = plt.subplots(6, n_subplots, figsize=(n_subplots * 4, 12))
    for i in range(n_subplots):
        axs[0, i].imshow(normals[0, i])
        axs[1, i].imshow(mask[0, i], cmap='gray')
        axs[2, i].imshow(color_map[0, i])
        axs[3, i].imshow(canonical_color_map[0, i])
        axs[4, i].imshow(w[0, i])
    plt.tight_layout()
    for ax in axs.flatten():
        ax.axis('off')
    plt.savefig('tinkering/test_renderer.png')
    plt.show()

    import ipdb; ipdb.set_trace()
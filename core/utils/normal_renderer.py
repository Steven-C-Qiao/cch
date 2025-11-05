import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex
)
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams


class NormalShader(nn.Module):
    """Shader for rendering surface normals with alpha channel for masking."""
    
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        """
        Render textures with alpha channel for masking.
        
        Args:
            fragments: Rasterization fragments from the rasterizer
            meshes: Meshes to render
            **kwargs: Additional arguments (e.g., blend_params)
            
        Returns:
            images: (N, H, W, C+1) where C is number of texture channels,
                    last channel is alpha mask
        """
        # Sample textures and blend
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        
        # Create alpha mask from fragments
        alpha = (fragments.zbuf[..., 0] > -1).unsqueeze(-1).float()
        
        # Combine texture channels with alpha mask
        if texels.shape[-1] > 3:
            # For multi-channel textures, concatenate with alpha
            images = torch.cat([texels, alpha], dim=-1)
        else:
            # For RGB textures, use standard blending then add alpha
            images_rgb = hard_rgb_blend(texels, fragments, blend_params)
            images = torch.cat([images_rgb, alpha], dim=-1)
        
        return images.squeeze()


class SurfaceNormalRenderer(pl.LightningModule):
    """
    Cleaned up renderer for surface normals from meshes.
    
    This renderer computes surface normals from mesh vertices and faces,
    optionally transforms them to camera space, and renders them as images.
    """
    
    def __init__(self, image_size=(256, 192)):
        """
        Args:
            image_size: Tuple of (height, width) for rendered images
        """
        super().__init__()
        self.image_size = image_size

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=160000 # Otherwise overflows 
        )
        
        
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=NormalShader()
        )
        
        self._set_cameras(PerspectiveCameras().to(self.device))

    def forward(self, mesh, **kwargs):
        """
        Render surface normals from mesh.
        
        Args:
            mesh: Meshes object (can be batched) - input mesh(es)
            **kwargs: Additional arguments (e.g., return_normalized)
                - return_normalized: bool - if True, return normals in [0, 1] range,
                                          if False, return normals in [-1, 1] range
                                      
        Returns:
            dict with keys:
                - 'normals': (N, H, W, 3) - rendered normal maps
                - 'masks': (N, H, W, 1) - alpha masks
        """
        return_normalized = kwargs.get('return_normalized', False)
        
        # Compute surface normals from mesh - use list to maintain proper alignment
        # verts_normals_list() returns a tuple, convert to list for item assignment
        surface_normals_list = list(mesh.verts_normals_list())  # List of (V_i, 3) tensors
        
        # Transform normals to camera space if cameras are set
        cameras = self.renderer.rasterizer.cameras
        if cameras is not None:
            # Extract R and T from cameras for normal transformation
            if hasattr(cameras, 'R') and hasattr(cameras, 'T'):
                R = cameras.R  # (N, 3, 3) tensor
                T = cameras.T  # (N, 3) tensor
                
                # Transform each mesh's normals to camera space: R^T @ normals
                for i, normals in enumerate(surface_normals_list):
                    # Transform normals: R^T @ normals
                    R_i = R[i]  # (3, 3)
                    normals_T = normals.T  # (3, V_i)
                    transformed = torch.mm(R_i.T, normals_T)  # (3, V_i)
                    surface_normals_list[i] = transformed.T  # (V_i, 3)
                    
                    # Invert z-axis for OpenCV convention
                    surface_normals_list[i][..., 2] *= -1
        
        # Normalize normals to [0, 1] for rendering (required by texture format)
        surface_normals_normalized_list = [(normals + 1.0) / 2.0 for normals in surface_normals_list]
        
        # Set mesh textures to normals using list - TexturesVertex handles list format properly
        mesh.textures = TexturesVertex(verts_features=surface_normals_normalized_list)
        
        # Render
        images = self.renderer(mesh)  # (N, H, W, 4) - last channel is alpha
        
        # Extract normals and masks
        normals = images[..., :3]  # (N, H, W, 3)
        masks = images[..., 3:4]  # (N, H, W, 1)
        
        # Denormalize back to [-1, 1] if requested
        if not return_normalized:
            normals = normals * 2.0 - 1.0
        
        return {
            'normals': normals,
            'masks': masks
        }
    
    def _set_cameras(self, cameras):
        """Set cameras for rendering.
        
        Args:
            cameras: Camera object (PerspectiveCameras or FoVPerspectiveCameras)
        """
        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras


import torch
import torch.nn as nn
import numpy as np


import sys 
sys.path.append('.')


# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)

from pytorch3d.renderer.cameras import PerspectiveCameras




class PointCloudRenderer(nn.Module):
    def __init__(self,
                 device,
                 batch_size,
                 f,
                 img_wh = 256,
                 cam_R = None,
                 cam_t = None, 
                 ):
        super().__init__()

        if cam_R is None:
            cam_R = torch.tensor([[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]], device=device).float()
            cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
        if cam_t is None:
            cam_t = torch.tensor([0., 0., 2.0]).float().to(device)[None, :].expand(batch_size, -1)

        self.cameras = PerspectiveCameras(
            device=device,
            R = cam_R,
            T = cam_t,
            focal_length=f,
            principal_point=((img_wh / 2., img_wh / 2.),),
            image_size=((img_wh, img_wh),),
            in_ndc=False
        )

        raster_settings = PointsRasterizationSettings(
            image_size=img_wh,
            radius=0.005, #0.003,
            points_per_pixel=10
        )

        self.rasteriser = PointsRasterizer(
            cameras=self.cameras,
            raster_settings=raster_settings
        )

        self.compositor = AlphaCompositor()

        self.renderer = PointsRenderer(
            rasterizer=self.rasteriser,
            compositor=AlphaCompositor()
        )

    def forward(self,
                point_clouds,
                cam_R=None,
                cam_t=None,
                f=None):
        if cam_t is not None:
            self.cameras.T = cam_t 
        if cam_R is not None:
            self.cameras.R = cam_R
        if f is not None:
            self.cameras.focal_length = f

        point_fragments = self.rasteriser(point_clouds)
        zbuffers = point_fragments.zbuf[:,:,:,0]

        # images = self.compositor(point_fragments, 
        #                          torch.ones_like(point_clouds))
        images = self.renderer(point_clouds)


        output = {
            'images': images,
            'depth_images': zbuffers
        }

        return output 


if __name__=="__main__":
    num_points = 1000
    points_x = torch.cos(torch.linspace(0, 1, num_points) * torch.pi * 2)
    points_y = torch.sin(torch.linspace(0, 1, num_points) * torch.pi * 2)
    points_z = torch.linspace(0,3,num_points)
    points = torch.stack([points_x, points_y, points_z], dim=-1)[None, ...]
    colors = torch.zeros([1, num_points, 3])
    colors[:, :, 2] = 1 # color red 

    point_cloud = Pointclouds(points, features=colors) 


    renderer = PointCloudRenderer(
        device='cpu',
        batch_size=1,
        f=200
    )
    

    renderer_output = renderer(point_cloud)


    rgb = (renderer_output['images'][0].contiguous().cpu().detach().numpy() * 255.).astype('uint8')  # (B, img_wh, img_wh, 3)
    depth = renderer_output['depth_images'][0].cpu().detach().numpy()


    import cv2 
    import matplotlib.pyplot as plt
    for i in range(1):
        cv2.imwrite(f"point_rend{i}.jpg", img=rgb)
        plt.imshow(depth)
        plt.colorbar()
        plt.savefig(f"point_rend{i}_depth.jpg")
        
    import ipdb 
    ipdb.set_trace()
    print('')
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

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

from pytorch3d.renderer.cameras import FoVPerspectiveCameras




class PointCloudRenderer(pl.LightningModule):
    def __init__(self,
                 image_size=(256, 192),
                 ):
        super().__init__()

        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.01, 
            points_per_pixel=10
        )

        self.rasteriser = PointsRasterizer(
            raster_settings=raster_settings
        )

        self.renderer = PointsRenderer(
            rasterizer=self.rasteriser,
            compositor=AlphaCompositor()#(background_color=torch.tensor([1, 1, 1]))
        )

    def forward(self,
                point_clouds,
                R=None,
                T=None):

        B, N = R.shape[:2]
        R = R.view(B * N, 3, 3)
        T = T.view(B * N, 3)
        if R is not None and T is not None:
            cameras = FoVPerspectiveCameras(R=R, T=T).to(self.device)
            self.renderer.rasterizer.cameras = cameras
            # self.renderer.cameras = cameras

        # Rasterize to get fragments (including z-buffer)
        fragments = self.rasteriser(point_clouds)
        zbuf = fragments.zbuf[..., 0]  # (BN, H, W)
        background_mask = (zbuf == -1)

        images = self.renderer(point_clouds)

        output = {
            'images': images,
            'background_mask': background_mask,
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
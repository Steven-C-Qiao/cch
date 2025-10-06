import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import os 
from core.configs.paths import BASE_PATH

from loguru import logger


def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

class SapiensWrapper(nn.Module):
    def __init__(
        self,
        project_dim=512,
        interpolate_size=(16, 16)
    ):
        super().__init__()
        self.model = self._build_sapiens()

        self.project_dim = project_dim
        self.interpolate_size = interpolate_size

        self._freeze()
        
        self.project = nn.Conv2d(1024, project_dim, kernel_size=1, stride=1, padding=0)
        # self.interpolate = F.interpolate(size=interpolate_size, mode='bilinear', align_corners=False)


    @staticmethod
    def _build_sapiens():
        model = load_model(os.path.join(BASE_PATH, 'model_files/sapiens_0.3b_epoch_1600_torchscript.pt2'), 
                           use_torchscript=True)
        dtype = torch.float32  # TorchScript models use float32
        model = model.cuda()
        return model

    def _freeze(self):
        logger.info(f"======== Freezing Sapiens Model ========")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    # @torch.compile
    def forward(self, image: torch.Tensor):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (out_local,) = self.model(image)

        out_local = self.project(out_local)
        out_local = F.interpolate(out_local, size=self.interpolate_size, mode='bilinear', align_corners=False).flatten(-2, -1)

        # Prepend some zero tokens for camera and register 
        out_local = torch.cat([torch.zeros_like(out_local[:, :, :5]), out_local], dim=-1)

        out_local = rearrange(out_local, 'b c p -> b p c') # p as num_of_tokens 

        return out_local


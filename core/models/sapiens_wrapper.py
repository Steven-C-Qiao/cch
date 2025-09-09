import torch
import torch.nn as nn

from loguru import logger


def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

class SapiensWrapper(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = self._build_sapiens()

        self._freeze()


    @staticmethod
    def _build_sapiens():
        model = load_model('/scratches/kyuban/cq244/CCH/cch/model_files/sapiens_0.3b_epoch_1600_torchscript.pt2', 
                           use_torchscript=True)
        dtype = torch.float32  # TorchScript models use float32
        model = model.cuda()
        return model

    def _freeze(self):
        logger.info(f"======== Freezing Sapiens Model ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    # @torch.compile
    def forward(self, image: torch.Tensor):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (out_local,) = self.model(image)

        return out_local


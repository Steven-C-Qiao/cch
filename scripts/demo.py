import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2
import argparse

from loguru import logger
import matplotlib.pyplot as plt

import sys 
sys.path.append('.')

from core.models.cch import CCH
from core.models.smpl import SMPL
from core.configs import paths
from core.configs.cch_cfg import get_cch_cfg_defaults
sapiens_ckpt_path = "/scratches/kyuban/cq244/CCH/sapiens/torchscript/normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
cch_ckpt_path = "/scratches/kyuban/cq244/CCH/cch/exp/exp_022_norm/saved_models/val_loss_epoch=051.ckpt"
img_dir_path = "/scratches/kyuban/cq244/CCH/cch/demo/demo_002/images"


class Config:
    CHECKPOINTS_DIR = "/scratches/kyuban/cq244/CCH/sapiens/torchscript/"
    CHECKPOINTS = {
        # "0.3b": "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
        # "0.6b": "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2",
        "1b": "normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
        # "2b": "sapiens_2b_normal_render_people_epoch_70_torchscript.pt2",
    }
    SEG_CHECKPOINTS = {
        "fg-bg-1b": "sapiens_1b_seg_foreground_epoch_8_torchscript.pt2",
        "no-bg-removal": None,
        "part-seg-1b": "seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    }


class ModelManager:
    @staticmethod
    def load_model(checkpoint_name, device):
        if checkpoint_name is None:
            return None
        checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to(device)
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        return F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)


class ImageProcessor:
    def __init__(self, device):

        self.mean = [123.5 / 255, 116.5 / 255, 103.5 / 255]
        self.std = [58.5 / 255, 57.0 / 255, 57.5 / 255]

        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.device = device

    def process_image(self, image: Image.Image, normal_model_name: str, seg_model_name: str):

        # Load models here instead of storing them as class attributes
        normal_model = ModelManager.load_model(Config.CHECKPOINTS[normal_model_name], self.device)
        input_tensor = self.transform_fn(image).unsqueeze(0).to(self.device)

        # Run normal estimation
        normal_map = ModelManager.run_model(normal_model, input_tensor, image.height, image.width)

        # Run segmentation
        if seg_model_name != "no-bg-removal":
            seg_model = ModelManager.load_model(Config.SEG_CHECKPOINTS[seg_model_name], self.device)
            seg_output = ModelManager.run_model(seg_model, input_tensor, image.height, image.width)
            seg_mask = (seg_output.argmax(dim=1) > 0).unsqueeze(0).repeat(1, 3, 1, 1)

        # Normalize and visualize normal map
        normal_map_norm = torch.linalg.norm(normal_map, dim=1, keepdim=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)
        normal_map_normalized[seg_mask == 0] = 1.0
        normal_map_normalized = normal_map_normalized.to(self.device)

        return normal_map_normalized, seg_mask

# class SapiensNormalPredictor:
#     def __init__(self, device):

#         self.mean = [123.5 / 255, 116.5 / 255, 103.5 / 255]
#         self.std = [58.5 / 255, 57.0 / 255, 57.5 / 255]

#         self.transform_fn = transforms.Compose([
#             transforms.Resize((1024, 768)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=self.mean, std=self.std),
#         ])
#         self.device = device

        
#         self.model = torch.jit.load(sapiens_ckpt_path)
#         self.model.eval()
#         self.model.to(self.device)

#     def process_image(self, image: Image.Image):
#         input_tensor = self.transform_fn(image).unsqueeze(0).to(self.device)
#         output = self.model(input_tensor)

#         ret = F.interpolate(output, size=(image.height, image.width), mode="bilinear", align_corners=False)
#         return ret


def main(sapiens_normal_model, cch_model, img_dir_path):
    with torch.no_grad():
        img_fnames = sorted(os.listdir(img_dir_path))
        print(f"Processing normal images")

        surface_normals = []
        seg_masks = []
        images = []

        for img_fname in tqdm(img_fnames):
            image_path = os.path.join(img_dir_path, img_fname)
            image = Image.open(image_path)
            
            # Handle EXIF orientation
            try:
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274)  # 274 is the EXIF tag for orientation
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # Image doesn't have EXIF data
                pass
                
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            normal_map, seg_mask = sapiens_normal_model.process_image(image, "1b", "part-seg-1b") # (1, 3, H, W)

            # Convert seg_mask to numpy for finding bounding box
            seg_mask_np = seg_mask.squeeze().cpu().numpy()[0]
            
            # Find bounding box of the human
            rows = np.any(seg_mask_np, axis=1)
            cols = np.any(seg_mask_np, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # Add some padding around the bounding box (10% of the box size)
            height, width = seg_mask_np.shape
            pad_y = int((y_max - y_min) * 0.1)
            pad_x = int((x_max - x_min) * 0.1)
            
            y_min = max(0, y_min - pad_y)
            y_max = min(height, y_max + pad_y)
            x_min = max(0, x_min - pad_x)
            x_max = min(width, x_max + pad_x)
            
            # Crop the image and normal map
            image = image.crop((x_min, y_min, x_max, y_max))
            normal_map = normal_map[:, :, y_min:y_max, x_min:x_max]
            seg_mask = seg_mask[:, :, y_min:y_max, x_min:x_max]
            
            # Pad to make square
            h, w = y_max - y_min, x_max - x_min
            if h > w:
                pad_left = (h - w) // 2
                pad_right = h - w - pad_left
                image = transforms.Pad((pad_left, 0, pad_right, 0), fill=0)(image)
                normal_map = F.pad(normal_map, (pad_left, pad_right, 0, 0), mode='replicate')
                seg_mask = F.pad(seg_mask, (pad_left, pad_right, 0, 0), mode='constant', value=0)
            elif w > h:
                pad_top = (w - h) // 2
                pad_bottom = w - h - pad_top
                image = transforms.Pad((0, pad_top, 0, pad_bottom), fill=0)(image)
                normal_map = F.pad(normal_map, (0, 0, pad_top, pad_bottom), mode='replicate')
                seg_mask = F.pad(seg_mask, (0, 0, pad_top, pad_bottom), mode='constant', value=0)

            image = image.resize((224, 224), Image.BILINEAR)
            normal_map = F.interpolate(normal_map, size=(224, 224), mode='bilinear', align_corners=False)
            seg_mask = F.interpolate(seg_mask.float(), size=(224, 224), mode='nearest').bool()

            #invert x color
            normal_map[:, 0, ...] *= -1

            normal_map = (((normal_map + 1) / 2)).squeeze()
            normal_map[seg_mask.squeeze() == 0] = 1.0

            surface_normals.append(normal_map) # as per my renderer convention [0, 1]
            seg_masks.append(seg_mask.squeeze())
            images.append(np.array(image))


            normal_map = (normal_map * 255).cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
            normal_map = normal_map[:, :, ::-1]

            # Create save directory
            save_dir = os.path.join(os.path.dirname(img_dir_path), "normal_maps")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"{os.path.splitext(img_fname)[0]}_normal.png")

            vis_image = np.concatenate([np.array(image), normal_map], axis=1)
            cv2.imwrite(save_path, vis_image)


        surface_normals = torch.stack(surface_normals).to(device)
        seg_masks = torch.stack(seg_masks).to(device)
        # Calculate padding amount to make width match height
        # _, _, H, W = surface_normals.shape
        # pad_amount = (H - W) // 2
        # pad_left = pad_amount
        # pad_right = H - W - pad_left

        print(f'surface_normals.shape: {surface_normals.shape}, seg_masks.shape: {seg_masks.shape}')
        
        # # Pad width to match height
        # surface_normals = F.pad(surface_normals, (pad_left, pad_right, 0, 0), mode='replicate')
        # seg_masks = F.pad(seg_masks, (pad_left, pad_right, 0, 0), mode='constant', value=0)
        
        # Resize to 224x224
        # surface_normals = F.interpolate(surface_normals, size=(224, 224), mode='bilinear', align_corners=False)
        # seg_masks = F.interpolate(seg_masks.float(), size=(224, 224), mode='nearest').bool()

        cch_output = cch_model(surface_normals)

        vc = cch_output['vc'].squeeze() # (4, 224, 224, 3)
        conf = cch_output['vc_conf'].squeeze() # (4, 224, 224)

        conf = conf.cpu().detach().numpy()

        conf_threshold = 0.08
        conf = 1 / conf
        conf_mask = (conf) < conf_threshold



        seg_masks = seg_masks.permute(0, 2, 3, 1).cpu().detach().numpy()[..., 0]
        vc = vc.cpu().detach().numpy()
        surface_normals = surface_normals.permute(0, 2, 3, 1).cpu().detach().numpy()



        # normalise 
        mask = seg_masks 

        aux_mask = seg_masks & conf_mask
        vc_copy = vc.copy()
        vc_copy[~aux_mask] = 0
        norm_min, norm_max = vc_copy.min(), vc_copy.max()


        # 3d scatter plot for vc
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vc_to_scatter = vc.reshape(-1, 3)[aux_mask.flatten()]
        ax.scatter(vc_to_scatter[..., 0], vc_to_scatter[..., 1], vc_to_scatter[..., 2], s=0.1, alpha=0.5)


        ax.view_init(elev=10, azim=20, vertical_axis='y')
        ax.set_box_aspect([1, 1, 1])
    
        max_range = np.array([
            vc_to_scatter[..., 0].max() - vc_to_scatter[..., 0].min(),
            vc_to_scatter[..., 1].max() - vc_to_scatter[..., 1].min(),
            vc_to_scatter[..., 2].max() - vc_to_scatter[..., 2].min()
        ]).max() / 2.0
        mid_x = (vc_to_scatter[..., 0].max() + vc_to_scatter[..., 0].min()) * 0.5
        mid_y = (vc_to_scatter[..., 1].max() + vc_to_scatter[..., 1].min()) * 0.5
        mid_z = (vc_to_scatter[..., 2].max() + vc_to_scatter[..., 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # plt.show()

        save_dir = os.path.join(os.path.dirname(img_dir_path), "vis")
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(os.path.join(save_dir, "vc_scatter.png"))
        # plt.show()

        plt.close()



        vc = (vc - norm_min) / (norm_max - norm_min)
        vc = np.clip(vc, 0, 1)
        vc[~mask] = 1

        conf[~mask] = 0

        fig = plt.figure(figsize=(4*4, 4*4))

        colors = [(1, 1, 1), (1, 0.5, 0)]  # White to orange
        import matplotlib
        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors)

        for i in range(4):
            plt.subplot(4, 4, i+1)
            plt.imshow(images[i])
            plt.subplot(4, 4, i+5)
            plt.imshow(vc[i])
            plt.subplot(4, 4, i+9)
            plt.imshow(surface_normals[i])
            plt.subplot(4, 4, i+13)
            im = plt.imshow(conf[i], cmap=custom_cmap)
            if i == 15:
                plt.colorbar(im)
        plt.savefig(os.path.join(save_dir, "colormaps.png"))
        import ipdb; ipdb.set_trace()
        print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_dir', 
        '-E', 
        type=str,
        help='Path to directory where logs and checkpoints are saved.'
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default='0,1', 
        help="Comma-separated list of GPU indices to use. E.g., '0,1,2'"
    )    
    parser.add_argument(
        "--dev", 
        action="store_true"
    )  
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {device}')

    device_ids = list(map(int, args.gpus.split(",")))
    logger.info(f"Using GPUs: {args.gpus} (Device IDs: {device_ids})")

    smpl_model = SMPL(
        model_path=paths.SMPL,
        num_betas=10,
        gender='neutral'
    )
    

    sapiens_model = ImageProcessor(device)


    cfg = get_cch_cfg_defaults()
    cch_model = CCH(cfg, smpl_model)
    ckpt = torch.load(cch_ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    # remove 'model' from keys
    ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
    cch_model.load_state_dict(ckpt, strict=False)
    cch_model.eval()
    cch_model.to(device)

    main(sapiens_model, cch_model, img_dir_path)

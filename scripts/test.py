import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2



class Config:
    CHECKPOINTS_DIR = "/scratch/cq244/Sculpture/model_files" #os.path.join(ASSETS_DIR, "checkpoints")
    CHECKPOINTS = {
        "1b": "/scratches/kyuban/cq244/CCH/sapiens/torchscript/normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2" # "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
    }
    SEG_CHECKPOINTS = {
        "fg-bg-1b": "sapiens_1b_seg_foreground_epoch_8_torchscript.pt2",
    }
    PART_SEG_CHECKPOINTS = {
        "1b": "/scratches/kyuban/cq244/CCH/sapiens/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
    }


class SapiensNormal:
    def __init__(self, device):

        self.mean = [123.5 / 255, 116.5 / 255, 103.5 / 255]
        self.std = [58.5 / 255, 57.0 / 255, 57.5 / 255]

        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.device = device

        ckpt_path = "/scratches/kyuban/cq244/CCH/sapiens/torchscript/normal/checkpoints/sapiens_1b/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
        self.model = torch.jit.load(ckpt_path)
        self.model.eval()
        self.model.to(self.device)

    def process_image(self, image: Image.Image):
        input_tensor = self.transform_fn(image).unsqueeze(0).to(self.device)
        output = self.model(input_tensor)

        ret = F.interpolate(output, size=(image.height, image.width), mode="bilinear", align_corners=False)
        return ret

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

    def process_image(self, image: Image.Image, part_seg_model_name: str, seg_model_name: str):

        # Load models here instead of storing them as class attributes
        part_seg_model = ModelManager.load_model(Config.PART_SEG_CHECKPOINTS[part_seg_model_name], self.device)
        input_tensor = self.transform_fn(image).unsqueeze(0).to(self.device)

        # Run part segmentation
        part_seg_logits = ModelManager.run_model(part_seg_model, input_tensor, image.height, image.width)

        # Run bg segmentation
        if seg_model_name != "no-bg-removal":
            seg_model = ModelManager.load_model(Config.SEG_CHECKPOINTS[seg_model_name], self.device)
            seg_output = ModelManager.run_model(seg_model, input_tensor, image.height, image.width)
            # seg_mask = (seg_output.argmax(dim=1) > 0).unsqueeze(0).repeat(1, 3, 1, 1)

        # import ipdb; ipdb.set_trace()
        # part_seg_logits[seg_mask == 0] = 0.0
        # part_seg_logits = part_seg_logits.to(self.device)

        return part_seg_logits, seg_output


def main(segmentor, path_to_dir):
    # Get all subdirectories in the path
    # subdirs = [d for d in os.listdir(path_to_dir) if os.path.isdir(os.path.join(path_to_dir, d))]
    
    # Sort subdirectories numerically
    # subdirs.sort(key=lambda x: int(x))
    
    subdirs = np.arange(1003, 1096)

    for subdir in tqdm(subdirs):
        subdir = str(subdir)
        opt1_subdir_path = os.path.join(path_to_dir, subdir, f"{subdir}_3Dapp/{subdir}")
        opt2_subdir_path = os.path.join(path_to_dir, subdir, f"ID_3Dapp/{subdir}")
        subdir_path = opt1_subdir_path if os.path.exists(opt1_subdir_path) else opt2_subdir_path
        # if subdir != "1006":
        #     continue


        for i in range(4):
            image_path = os.path.join(subdir_path, f"0{i}.png")
            # try:
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            else:
                assert False, f"Image {image_path} is not RGBA"

            print(f"Processing image {image_path}")

            part_seg_output, seg_output = segmentor.process_image(image, "1b", "fg-bg-1b")

            # Use part_seg output as mask, bg_model is not stable
            part_seg = part_seg_output.argmax(dim=1).squeeze(0).cpu().detach().numpy()
            part_seg = (part_seg > 1).astype(np.uint8)[..., None]

            masked_img = np.array(image) * part_seg


            save_dir = os.path.join(path_to_dir, subdir, f"{subdir}_segmented")
            os.makedirs(save_dir, exist_ok=True)

            cv2.imwrite(os.path.join(save_dir, f"{subdir}_0{i}.png"), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

            # import matplotlib.pyplot as plt
            # plt.imshow(masked_img)
            # plt.show()
            # import ipdb; ipdb.set_trace()

            # mask = seg_output.clone()[0, 1] # foreground logits
            # # set mask to 1 if it's in the top 20%, 0 otherwise
            # mask = (mask > mask.quantile(0.825)).float()
            

            # # seg_mask is 1, 2, 768, 768
            # import matplotlib.pyplot as plt
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            # im1 = ax1.imshow(seg_output.squeeze(0)[0].cpu().detach().numpy(), label="channel 1")
            # im2 = ax2.imshow(seg_output.squeeze(0)[1].cpu().detach().numpy(), label="channel 2")
            # im3 = ax3.imshow(mask.squeeze().cpu().detach().numpy(), label="seg_output")
            # # im3 = ax3.imshow(part_seg_output.squeeze(0).cpu().detach().numpy(), label="part_seg_output")
            # fig.colorbar(im1, ax=ax1)
            # fig.colorbar(im2, ax=ax2)
            # fig.colorbar(im3, ax=ax3)
            # plt.show()
            # plt.close()
            # import ipdb; ipdb.set_trace()



            # seg_mask = (seg_mask.squeeze(0).cpu().detach().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            # cv2.imwrite(os.path.join(save_dir, f"{subdir}_0{i}_seg_mask.png"), seg_mask)


if __name__ == "__main__":
    model_name = "fg-bg-1b"
    # path_to_dir = "/scratch/cq244/Sculpture/demo/Addinbrokes_3DO_substudy"
    path_to_dir = "/scratches/kyuban/cq244/datasets/3DO_substudy/3DO"

    # save_dir = "/scratch/cq244/Sculpture/demo/Addinbrokes_3DO_substudy_segmented"
    # os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentor = ImageProcessor(device)
    main(segmentor, path_to_dir)

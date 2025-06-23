import numpy as np
import torch
import sys 
sys.path.append('.')
from core.utils.renderer import SurfaceNormalRenderer
from core.utils.sample_utils import sample_cameras

from core.configs.cch_cfg import get_cch_cfg_defaults



BETA_PATH = '/scratches/kyuban/cq244/datasets/CAPE/minimal_body_shape/00134/00134_param.pkl'
PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00134/longlong_ballet1_trial1'
FIRST_FRAME_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00134/longlong_ballet1_trial1/longlong_ballet1_trial1.000001.npz'

def load_single_frame(npz_fn):
    '''
    given path to a single data frame, return the contents in the data
    '''
    data = np.load(npz_fn)
    return data['v_cano'], data['v_posed'], data['pose'], data['transl']


if __name__ == "__main__":
    renderer = SurfaceNormalRenderer(
        image_size=(224, 224)
    )
    cfg = get_cch_cfg_defaults()

    sample_frames = [
        'longlong_ballet1_trial1.000050.npz',
        'longlong_ballet1_trial1.000100.npz',
        'longlong_ballet1_trial1.000150.npz',
        'longlong_ballet1_trial1.000200.npz',
    ]

    first_frame_v_cano, first_frame_v_posed, first_frame_pose, first_frame_transl = load_single_frame(FIRST_FRAME_PATH)

    print(first_frame_v_posed.shape)



    R, T = sample_cameras(1, 4, cfg.DATA)






    for frame in sample_frames:
        v_cano, v_posed, pose, transl = load_single_frame(f'{PATH}/{frame}')


        first_frame_render_ret = renderer(
            vertices = torch.tensor(v_posed)[None, None].repeat(1, 4, 1, 1).float(), 
            R=R, 
            T=T, 
            first_frame_v_cano=torch.tensor(first_frame_v_cano)[None].float())
        
        first_frame_pm = first_frame_render_ret['canonical_color_maps']

        render_ret = renderer(
            vertices = torch.tensor(v_posed)[None, None].repeat(1, 4, 1, 1).float(), 
            R=R, 
            T=T, 
            first_frame_v_cano=torch.tensor(v_cano)[None].float())
        
        pm = render_ret['canonical_color_maps']

        diff_pm = first_frame_pm - pm
        print(diff_pm.max(), diff_pm.min())
        
        # Compute the norm of the difference
        diff_pm_norm = torch.norm(diff_pm, dim=-1)  # (B, N, H, W)
        print(f"diff_pm_norm max: {diff_pm_norm.max()}, min: {diff_pm_norm.min()}")

        break

    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Create figure with 3 rows
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))

        

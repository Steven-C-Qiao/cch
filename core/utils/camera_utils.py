import torch
import numpy as np

def uv_to_pixel_space(intrinisics):
    # Undo the UV space transformation by inverting the uv_intrinsic matrix from load_calib
    pixel_intrinsics = np.copy(intrinisics)
    render_size = 512  # Same as default in load_calib
    
    # Multiply by render_size/2 to convert from UV coordinates back to pixel coordinates
    pixel_intrinsics[0, 0] *= float(render_size // 2)
    pixel_intrinsics[1, 1] *= float(render_size // 2) 
    pixel_intrinsics[2, 2] *= float(render_size // 2)
    
    return pixel_intrinsics

def custom_opengl2pytorch3d(extrinsic):
    transl = extrinsic[:3, 3]

    # Flip x and z axis
    coord_convert = torch.tensor([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],  
        [0, 0, 0, 1]
    ]).float()

    extrinsic = coord_convert @ extrinsic

    R = extrinsic[:3, :3]

    transl[[0, -1]] *= -1

    # mistake during rendering saving extrinsics
    # undo here 
    transl = -R.T @ transl

    transl[[1]] *= -1

    R = R.T

    return R, transl
    
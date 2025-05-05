import torch
from pytorch3d.renderer.cameras import look_at_view_transform

def sample_cameras(batch_size, num_views, t):
    azim = torch.rand(batch_size * num_views) * 360
    elev = torch.rand(batch_size * num_views) * 20 - 10
    dist = torch.rand(batch_size * num_views) * 2 + 1.2
    t = t.repeat_interleave(num_views, dim=0)

    R, T = look_at_view_transform(dist=dist, 
                                    elev=elev, 
                                    azim=azim, 
                                    at=t,
                                    degrees=True)
    R = R.view(batch_size, num_views, 3, 3)
    T = T.view(batch_size, num_views, 3)
    return R.to(t), T.to(t)
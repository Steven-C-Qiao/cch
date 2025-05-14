import torch
from pytorch3d.renderer.cameras import look_at_view_transform

def sample_cameras(batch_size, num_views, t=None):
    azim = torch.tensor([[0, 90, 180, 270]])
    azim_noise = torch.rand(batch_size, num_views) * 30 - 15
    azim = (azim + azim_noise).view(-1)

    elev = torch.rand(batch_size * num_views) * 20 - 10
    dist = torch.randn(batch_size * num_views) * 0.2 + 2.5
    if t is not None:
        t = t.view(-1, 3)

    R, T = look_at_view_transform(  dist=dist, 
                                    elev=elev, 
                                    azim=azim, 
                                    # at=t,
                                    degrees=True)
    R = R.view(batch_size, num_views, 3, 3)
    T = T.view(batch_size, num_views, 3)
    return R, T
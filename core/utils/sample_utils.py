import torch
from pytorch3d.renderer.cameras import look_at_view_transform

def sample_cameras(batch_size, num_views, cfg):
    azim = torch.tensor([[0, 90, 180, 270]])
    azim_noise = torch.rand(batch_size, num_views) * 30 - 15
    azim = (azim + azim_noise).view(-1)
    azim = azim[torch.randperm(azim.shape[0])]

    # azim = torch.rand(batch_size * num_views) * 360

    elev = torch.randn(batch_size * num_views) * cfg.ELEV_STD
    dist = torch.randn(batch_size * num_views) * cfg.DIST_STD + cfg.DIST_MEAN

    at = torch.randn(batch_size * num_views, 3) * cfg.AT_STD

    R, T = look_at_view_transform(  dist=dist, 
                                    elev=elev, 
                                    azim=azim, 
                                    at=at,
                                    degrees=True)
    R = R.view(batch_size, num_views, 3, 3)
    T = T.view(batch_size, num_views, 3)
    return R, T
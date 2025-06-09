import torch
from pytorch3d.renderer.cameras import look_at_view_transform

def sample_cameras(batch_size, num_views, cfg):
    azim = torch.tensor([[0, 90, 180, 270]])
    azim_noise = torch.randn(batch_size, num_views) * 15
    azim = (azim + azim_noise).view(-1)
    
    # Randomly permute azimuth angles for each batch
    azim = azim.view(batch_size, num_views)
    perm = torch.randperm(num_views)
    azim = azim[:, perm].view(-1)

    elev = torch.randn(batch_size * num_views) * cfg.ELEV_STD + cfg.ELEV_OFFSET
    dist = torch.randn(batch_size * num_views) * cfg.DIST_STD + cfg.DIST_OFFSET

    at = torch.randn(batch_size * num_views, 3) * cfg.AT_STD + cfg.AT_OFFSET 

    R, T = look_at_view_transform(  dist=dist, 
                                    elev=elev, 
                                    azim=azim, 
                                    at=at,
                                    degrees=True)
    R = R.view(batch_size, num_views, 3, 3)
    T = T.view(batch_size, num_views, 3)
    return R, T
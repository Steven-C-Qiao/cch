import torch

class MetricsCalculator:
    def __init__(self):
        self.metrics = ['colormap_dist']

        self.total_num_sets = 0
        self.total_num_views = 0

        self.metrics_dict = {}
        for metric in self.metrics:
            self.metrics_dict[metric] = []

    def update(self, pred_dict, targets_dict, batch_size):
        
        num_views = pred_dict['sculpted_pred_dict']['vp'].shape[1]
        with torch.no_grad():
            if 'PVE' in self.metrics:
                PVE = torch.linalg.norm(pred_dict['sculpted_pred_dict']['vp'] - targets_dict['vp'], dim=-1)
                assert PVE.shape == (batch_size, num_views, 6890)
                PVE = torch.mean(PVE)
                self.metrics_dict['PVE'].append(PVE)

                HMR_PVE = torch.linalg.norm(pred_dict['hmr_pred_dict']['vp'] - targets_dict['vp'], dim=-1)
                assert HMR_PVE.shape == (batch_size, num_views, 6890)
                HMR_PVE = torch.mean(HMR_PVE)
                self.metrics_dict['HMR-PVE'].append(HMR_PVE)

            self.total_num_sets += batch_size
            self.total_num_views += batch_size * num_views


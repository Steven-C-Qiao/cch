import torch 
import os 
import trimesh 
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pytorch_lightning as pl 
import matplotlib.colors
# import scenepic as sp 
from einops import rearrange
from collections import defaultdict

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


try:
    import pyvista as pv
    PV_AVAILABLE = True
    # Set PyVista to use offscreen rendering
    pv.OFF_SCREEN = True
except ImportError:
    PV_AVAILABLE = False

# from core.losses.cch_loss import point_map_to_normal



IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class Visualiser(pl.LightningModule):
    def __init__(self, save_dir, cfg=None, rank=0):
        super().__init__()
        self.save_dir = save_dir
        self.rank = rank
        self.cfg = cfg
        self.threshold = cfg.LOSS.CONFIDENCE_THRESHOLD if cfg is not None else 100
        self.mask_percentage = cfg.LOSS.CONFIDENCE_MASK_PERCENTAGE if cfg is not None else 0.0
        self.num_frames_pp = getattr(cfg.DATA, 'NUM_FRAMES_PP', 4) if cfg is not None else 4
        self._suffix = ''

        self.counter = 0

        # self.threshold = 50
        # print("Visualiser confidence threshold:", self.threshold)

    def _get_confidence_threshold_from_percentage(self, confidence, image_mask):
        """
        Compute threshold value that masks a certain percentage of foreground pixels with lowest confidence.
        
        Args:
            confidence: Confidence values array (N, H, W) or (H, W), can be torch tensor or numpy array
            image_mask: Foreground mask (N, H, W) or (H, W), can be torch tensor or numpy array
            
        Returns:
            Threshold value to use for masking
        """
        if self.mask_percentage <= 0.0:
            return self.threshold
        
        # Convert to numpy if torch tensors
        if hasattr(confidence, 'cpu'):
            confidence = confidence.cpu().detach().numpy()
        if hasattr(image_mask, 'cpu'):
            image_mask = image_mask.cpu().detach().numpy()
        
        # Ensure mask is boolean
        image_mask = image_mask.astype(bool)
        
        # Flatten for easier processing
        confidence_flat = confidence.flatten()
        mask_flat = image_mask.flatten()
        
        # Get confidence values only for foreground pixels
        foreground_conf = confidence_flat[mask_flat]
        
        if len(foreground_conf) == 0:
            return self.threshold
        
        # Calculate the threshold value for the given percentage
        # We want to mask the lowest mask_percentage of foreground pixels
        percentile = self.mask_percentage * 100
        computed_threshold = np.percentile(foreground_conf, percentile)
        
        # Use the computed threshold
        return computed_threshold

    def set_global_rank(self, global_rank):
        self.rank = global_rank

    def _get_filename(self, suffix=''):
        """
        Generate filename with format: {counter:06d}_{epoch:03d}_{split}{suffix}.png
        """
        split_part = f'_{self._split}' if self._split else ''
        return f'{self.counter:06d}_{self._epoch:03d}{split_part}{suffix}.png'


    def visualise_debug_loss(self, loss_dict):
        debug_loss_pred2gt_conf = loss_dict['debug_loss_pred2gt_conf'].cpu().detach().numpy()
        debug_loss_pred2gt = loss_dict['debug_loss_pred2gt'].cpu().detach().numpy()
        # debug_loss_gt2pred = loss_dict['debug_loss_gt2pred'].cpu().detach().numpy()

        B, K, N, H, W = debug_loss_pred2gt_conf.shape
        B = 1
        N = min(N, 4)
        sub_fig_size = 4
        num_cols = N
        num_rows = 2 * K
        
        
        fig = plt.figure(figsize=(num_cols*sub_fig_size, num_rows*sub_fig_size))
        for k in range(K):
            for n in range(N):
                row = 2 * k
                ax = plt.subplot(num_rows, num_cols, row*num_cols + n + 1)
                # Plot conf in normal scale (can be negative)
                conf_data = debug_loss_pred2gt_conf[0, k, n].copy()
                # Mask zero values to be white by setting them to NaN
                # zero_mask = (conf_data == 0.0)
                # conf_data[zero_mask] = np.nan
                # # Also mask very large values to NaN for readability
                # high_mask = (conf_data > 10)
                # conf_data[high_mask] = np.nan
                # Use the same diverging colormap style as error visualisation
                im = ax.imshow(conf_data, cmap='RdYlGn_r')
                ax.set_title(f'Debug Loss Pred2GT Conf {k}, {n}')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                row = 2 * k + 1
                ax = plt.subplot(num_rows, num_cols, row*num_cols + n + 1)
                # Visualize loss with linear color scale
                loss_data = debug_loss_pred2gt[0, k, n].copy()
                # Mask zero values to be white by setting them to NaN
                zero_mask = (loss_data == 0.0)
                # loss_data[zero_mask] = np.nan
                # Also mask very large values to NaN for readability
                # high_mask = (loss_data > 10)
                # loss_data[high_mask] = np.nan
                im = ax.imshow(loss_data, cmap='RdYlGn_r')
                ax.set_title(f'Debug Loss Pred2GT log {k}, {n}')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

                # plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                # plt.imshow(debug_loss_gt2pred[0, k, n])
                # plt.title(f'Debug Loss GT2Pred {k}, {n}')
                # plt.colorbar()

        plt.tight_layout(pad = 0.1)
        plt.savefig(os.path.join(self.save_dir, f'debug_loss.png'), 
                    bbox_inches='tight', pad_inches=0.1)
        plt.close()

    
    def visualise_debug_vc_pm_loss(self, loss_dict):
        debug_vc_pm_loss_conf = loss_dict['debug_vc_pm_loss_conf'].cpu().detach().numpy()
        debug_vc_pm_loss = loss_dict['debug_vc_pm_loss'].cpu().detach().numpy()
        
        B, N, H, W = debug_vc_pm_loss_conf.shape
        B = 1
        N = min(N, self.num_frames_pp)
        sub_fig_size = 4
        num_cols = 4
        num_rows = 2
        
        
        fig = plt.figure(figsize=(num_cols*sub_fig_size, num_rows*sub_fig_size))
        for n in range(N):
            row = 0
            ax = plt.subplot(num_rows, num_cols, row*num_cols + n + 1)
            
            vc_loss_conf = debug_vc_pm_loss_conf[0, n].copy()

            zero_mask = (vc_loss_conf == 0.0)
            vc_loss_conf[zero_mask] = np.nan

            im = ax.imshow(vc_loss_conf, cmap='RdYlGn_r')
            ax.set_title(f'Debug VC PM Loss Conf {n}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            row = 1
            ax = plt.subplot(num_rows, num_cols, row*num_cols + n + 1)
            vc_loss = debug_vc_pm_loss[0, n].copy()
            zero_mask = (vc_loss == 0.0)
            vc_loss[zero_mask] = np.nan

            im = ax.imshow(vc_loss, cmap='RdYlGn_r')
            ax.set_title(f'Debug VC PM Loss {n}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        plt.tight_layout(pad = 0.1)
        plt.savefig(os.path.join(self.save_dir, f'debug_vc_pm_loss.png'), dpi=300,
                    bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def visualise(
        self, 
        predictions,
        batch,
        metrics=None,
        batch_idx=None,
        split=None,
        epoch=None
    ):
        """
        Args:
            K = N, pm_k^n is the k-th pose plotted on the n-th shape/silhouette
            normal_maps: (B, N, H, W, 3)

            vc: (B, N, H, W, 3)
            vc_init_pred: (B, N, H, W, 3): w/o pose blendshapes
            vc_pred: (B, K, N, H, W, 3)

            vp: (B, N, 6890, 3)
            vp_init_pred: (B, K, N, H, W, 3): w/o pose blendshapes
            vp_pred: (B, K, N, H, W, 3)

            dvc: (B, K, N, H, W, 3)
            dvc_pred: (B, K, N, H, W, 3)
            
            conf: (B, N, H, W)
            mask: (B, N, H, W)
            color: (B, N, H, W)
            vertex_visibility: (B, N, 6890)
        """
        if self.rank != 0:
            return None 
        
        # set suffix for this visualisation pass
        self._suffix = f"_{epoch}_{split}" if epoch is not None and split else ''
        # Store epoch and split separately for file naming
        self._epoch = epoch if epoch is not None else 0
        self._split = split if split else ''

        # if batch_idx is not None:
        #     self.counter = batch_idx
        # else:
        #     self.counter = self.global_step
        

        # if 'gt_normal_maps' in batch:
        #     B, K, H, W, C = batch['gt_normal_maps'].shape
        #     batch['gt_normal_maps'] = batch['gt_normal_maps'][:, :N] # b n h w c
        # if 'vp' in predictions:
        #     B, K, N, H, W, C = predictions['vp'].shape
        #     pred = predictions['vp']
        #     pred = torch.stack([pred[:, i, i] for i in range(4)], dim=1)
        #     mask = batch['masks'][:, :N]
        #     pred_normals, pred_valids = point_map_to_normal(pred, mask)

        #     pred_normals = pred_normals[pred_valids]

        #     predictions['vp_normal_maps'] = pred_normals
        


        # Convert predictions to numpy if tensor
        for k, v in predictions.items():
            predictions[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items():
            batch[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v

        if metrics is not None:
            for k, v in metrics.items():
                metrics[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v

        # self.visualise_input_normal_imgs(
        #     normal_maps
        # )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.visualise_initial_pms(
            predictions,
            batch
        )

        # self.visualise_pbs_pms(
        #     predictions,
        #     batch
        # )

        self.visualise_full_pyvista(
            predictions,
            batch,
            metrics
        )

        # self.visualise_full(
        #     predictions,
        #     batch
        # )

        # self.visualise_vp_vc(
        #     predictions,
        #     batch
        # )

        if torch.cuda.is_available():
            torch.cuda.synchronize()




    def visualise_initial_pms(
        self, 
        predictions,
        batch,
    ):
        B, N, H, W, C = predictions['vc_init'].shape
        image_masks = batch['masks']

        image_masks_N = image_masks[:, :N]

        smpl_masks = batch['smpl_mask']
        smpl_masks_N = smpl_masks[:, :N]

        mask_intersection = image_masks_N * smpl_masks_N
        mask_union = image_masks_N + smpl_masks_N - mask_intersection
        mask_union = mask_union.astype(bool)

        dataset_name = batch['dataset'][0]


        B = 1
        N = min(N, self.num_frames_pp)
        sub_fig_size = 4
        num_cols = N

        num_rows = 0

        if 'imgs' in batch:
            num_rows += 1
            images = rearrange(batch['imgs'][0], 'n c h w -> n h w c')
            # images = (images * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
            images = images.astype(np.float32)

        if 'vc_init' in predictions:
            num_rows += 1
            vc_init = predictions['vc_init']

            if 'vc_maps' in batch: # 4DDress
                num_rows += 1
                vc_init_err = np.linalg.norm(predictions['vc_init'] - batch['vc_maps'][:, :N], axis=-1)
                vc_init_err = vc_init_err * mask_intersection

            if 'vc_smpl_maps' in batch: # THuman
                num_rows += 1
                vc_init_err = np.linalg.norm(predictions['vc_init'] - batch['vc_smpl_maps'][:, :N], axis=-1)
                vc_init_err = vc_init_err * mask_intersection

            vc_init[~mask_union.astype(bool)] = 0

            norm_min, norm_max = vc_init.min(), vc_init.max()
            vc_init = (vc_init - norm_min) / (norm_max - norm_min) 
            vc_init[~mask_union.astype(bool)] = 1
        
        if 'vc_init_conf' in predictions:
            num_rows += 1
            vc_init_conf = predictions['vc_init_conf']
            vc_init_conf = vc_init_conf * mask_union 

        # 4DDress ground truth VC maps
        if 'vc_maps' in batch:
            num_rows += 1
            vc_maps = batch['vc_maps']
            temp_mask = batch['smpl_mask']
            vc_maps[~temp_mask.astype(bool)] = 0
            vc_maps = (vc_maps - vc_maps.min()) / (vc_maps.max() - vc_maps.min())
            vc_maps[~temp_mask.astype(bool)] = 1
        
        # THuman ground truth VC maps
        if 'vc_smpl_maps' in batch:
            num_rows += 1
            vc_maps = batch['vc_smpl_maps']
            temp_mask = batch['smpl_mask']
            vc_maps[~temp_mask.astype(bool)] = 0
            vc_maps = (vc_maps - vc_maps.min()) / (vc_maps.max() - vc_maps.min())
            vc_maps[~temp_mask.astype(bool)] = 1


        if 'smpl_w_maps' in batch:
            num_rows += 1
            smpl_w_maps = np.argmax(batch['smpl_w_maps'], axis=-1)

        if "w" in predictions:
            num_rows += 1
            w = predictions['w']
            w = np.argmax(w, axis=-1)
            w = w * image_masks_N 

    
        fig = plt.figure(figsize=(num_cols*sub_fig_size, num_rows*sub_fig_size))
        for n in range(num_cols):
            row = 0

            if 'imgs' in batch:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(images[n])
                plt.title(f'Image {n}')
                row += 1

            if 'vc_maps' in batch or 'vc_smpl_maps' in batch:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(vc_maps[0, n])
                plt.title(f'gt $V_{n+1}^c$')
                row += 1
                
            if 'vc_init' in predictions:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(vc_init[0, n])
                plt.title(f'Pred init $V_{n+1}^c$')
                row += 1

            if 'vc_maps' in batch or 'vc_smpl_maps' in batch:
                ax = plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                # Apply log scale to error visualization
                err_vis = vc_init_err[0, n].copy()
                # Mask zero errors to be white by setting them to NaN
                zero_mask = (err_vis == 0.0)
                err_vis[zero_mask] = np.nan
                # Use LogNorm so colorbar shows original values but visualization uses log scale
                # Find valid (non-NaN) error range for normalization
                valid_errors = err_vis[~np.isnan(err_vis)]
                if len(valid_errors) > 0:
                    vmin = np.min(valid_errors[valid_errors > 0])  # Smallest non-zero error
                    vmax = np.max(valid_errors)  # Largest error
                    # Use LogNorm with a small offset to handle very small values
                    norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, 1e-6))
                    im = plt.imshow(err_vis, cmap='RdYlGn_r', norm=norm)
                else:
                    # Fallback if no valid errors
                    im = plt.imshow(err_vis, cmap='RdYlGn_r')
                plt.title(f'Error $V_{n+1}^c$')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                row += 1

            if 'vc_init_conf' in predictions:
                ax = plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                # Apply log scale to confidence visualization
                conf_vis = vc_init_conf[0, n].copy()
                # Mask zero confidence to be white by setting them to NaN
                zero_mask = (conf_vis == 0.0)
                conf_vis[zero_mask] = np.nan
                # Use LogNorm so colorbar shows original values but visualization uses log scale
                # Find valid (non-NaN) confidence range for normalization
                valid_conf = conf_vis[~np.isnan(conf_vis)]
                if len(valid_conf) > 0:
                    vmin = np.min(valid_conf[valid_conf > 0])  # Smallest non-zero confidence
                    vmax = np.max(valid_conf)  # Largest confidence
                    # Use LogNorm with a small offset to handle very small values
                    norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, 1e-6))
                    im = plt.imshow(conf_vis, cmap='viridis', norm=norm)
                else:
                    # Fallback if no valid confidence
                    im = plt.imshow(conf_vis, cmap='viridis')
                plt.title(f'Conf $V_{n+1}^c$')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                row += 1

            if 'smpl_w_maps' in batch:
                ax = plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                im = plt.imshow(smpl_w_maps[0, n])
                plt.title(f'Smpl $w_{n+1}$')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                row += 1
            
            if "w" in predictions:
                ax = plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                im = plt.imshow(w[0, n])
                plt.title(f'Pred $w_{n+1}$')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                row += 1

            # # Ground truth normals
            # if gt_normal_maps is not None:
            #     plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
            #     plt.imshow(gt_normal_maps[0, n])
            #     plt.title(f'GT Normal $n_{n+1}$')
            #     row += 1
            
            # # Predicted normals from vc_init
            # if pred_normal_maps is not None:
            #     plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
            #     plt.imshow(pred_normal_maps[0, n])
            #     plt.title(f'Pred Normal $n_{n+1}$')
            #     row += 1

        # for ax in fig.axes:
        #     ax.set_xticks([])
        #     ax.set_yticks([])


        plt.tight_layout(pad = 0.1)
        plt.savefig(os.path.join(self.save_dir, self._get_filename(f'_{dataset_name}_vc')))
        plt.close()



    def visualise_pbs_pms(
        self,
        predictions,
        batch
    ):
        dataset_name = batch['dataset'][0]
        B, N, H, W, C = predictions['vc_init'].shape
        K = self.num_frames_pp + 1  # Total frames = num_frames_pp + 1
        image_masks = batch['masks']

        image_masks_N = image_masks[:, :N]

        smpl_masks = batch['smpl_mask']
        smpl_masks_N = smpl_masks[:, :N]

        mask_intersection = image_masks_N * smpl_masks_N
        mask_union = image_masks_N + smpl_masks_N - mask_intersection
        mask_union = mask_union.astype(bool)


        
        
        mask = np.repeat(batch['masks'][:, :N][:, None], K, axis=1) # B, K, N, H, W

        B = 1
        N = min(N, self.num_frames_pp)
        sub_fig_size = 4
        num_cols = N

        num_rows = 0

        if 'vp_init' in predictions:
            num_rows += K
            vp_init = predictions['vp_init'] # bknhwc
            vp_init[~mask.astype(bool)] = 0
            norm_min, norm_max = vp_init.min(), vp_init.max()
            vp_init = (vp_init - norm_min) / (norm_max - norm_min) 
            vp_init[~mask.astype(bool)] = 1


        if 'vc' in predictions:
            num_rows += K 
            vc = predictions['vc'] # bknhwc
            vc[~mask.astype(bool)] = 0
            norm_min, norm_max = vc.min(), vc.max()
            vc = (vc - norm_min) / (norm_max - norm_min) 
            vc[~mask.astype(bool)] = 1

        if 'vp' in predictions:
            num_rows += K 
            vp = predictions['vp'] # bknhwc
            vp[~mask.astype(bool)] = 0
            norm_min, norm_max = vp.min(), vp.max()
            vp = (vp - norm_min) / (norm_max - norm_min) 
            vp[~mask.astype(bool)] = 1

        if 'dvc' in predictions:
            num_rows += K 
            dvc = predictions['dvc'] # bknhwc
            dvc = np.linalg.norm(dvc, axis=-1)
            dvc = dvc * mask

        # Skip visualization if there's nothing to plot
        if num_rows == 0:
            return

        fig = plt.figure(figsize=(num_cols*sub_fig_size, num_rows*sub_fig_size))
        row = 0
        for k in range(K):
            if 'vp_init' in predictions:
                for n in range(N):
                    plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                    plt.imshow(vp_init[0, k, n])
                    plt.title(f'Pred init $V_{n+1}^{k+1}$')
                row += 1

            if 'dvc' in predictions:
                for n in range(N):
                    subplot_idx = (row)*num_cols + n + 1
                    if subplot_idx > num_rows * num_cols:
                        break
                    ax = plt.subplot(num_rows, num_cols, subplot_idx)
                    dvc_vis = dvc[0, k, n].copy()
                    # Mask background to nan
                    dvc_vis[~mask[0, k, n].astype(bool)] = np.nan
                    im = ax.imshow(dvc_vis)
                    plt.title(f'Pred $\\Delta V_{n+1}^{{c,{k+1}}}$')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
                row += 1

            if 'vc' in predictions:
                for n in range(N):
                    plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                    plt.imshow(vc[0, k, n])
                    plt.title(f'Pred $V_{n+1}^{{c,{k+1}}}$')
                row += 1

            if 'vp' in predictions:
                for n in range(N):
                    plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                    plt.imshow(vp[0, k, n])
                    plt.title(f'Pred final $V_{n+1}^{k+1}$')
                row += 1
            


        plt.tight_layout(pad = 0.1)
        plt.savefig(os.path.join(self.save_dir, self._get_filename(f'_{dataset_name}_pbs')))
        plt.close()




    def _no_annotations(self, fig):
        for ax in fig.axes:
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none') 
            ax.zaxis.pane.set_edgecolor('none')
            ax.xaxis.line.set_color('none')
            ax.yaxis.line.set_color('none')
            ax.zaxis.line.set_color('none')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])



    def visualise_full_pyvista(
        self,
        predictions,
        batch,
        metrics=None
    ):
        """
        Fast version of visualise_full using PyVista for point cloud rendering.
        Alternative to Open3D version.
        """
        if not PV_AVAILABLE:
            print("Warning: PyVista not available, falling back to matplotlib visualise_full")
            return self.visualise_full(predictions, batch)

        dataset_name = batch['dataset'][0]
        
        subfig_size = 4
        gt_alpha, pred_alpha = 0.5, 0.5
        point_size = 0.7 # PyVista point size

        B, N, H, W, C = predictions['vc_init'].shape
        K = self.num_frames_pp + 1  # Total frames = num_frames_pp + 1

        image_masks = batch['masks']
        image_masks_N = image_masks[:, :N]

        smpl_masks = batch['smpl_mask']
        smpl_masks_N = smpl_masks[:, :N]

        mask_intersection = image_masks_N * smpl_masks_N
        mask_union = image_masks_N + smpl_masks_N - mask_intersection
        mask_union = mask_union.astype(bool)


        scatter_mask = image_masks_N[0].astype(bool) # nhw
        if "vc_init_conf" in predictions:
            confidence = predictions['vc_init_conf']
            # Compute threshold from percentage if enabled, otherwise use fixed threshold
            threshold_value = self._get_confidence_threshold_from_percentage(
                confidence[0], image_masks_N[0]
            )
            confidence = confidence > threshold_value
        else:
            confidence = np.ones_like(predictions['vc_init'])[..., 0].astype(bool)


        scatter_mask = scatter_mask * confidence[0].astype(bool)
        
        # If scatter_mask is empty (all confidence below threshold), fall back to image masks only
        if not np.any(scatter_mask):
            scatter_mask = image_masks_N[0].astype(bool)
        
        # Color for predicted scatters 
        color = rearrange(batch['imgs'][0, :N], 'n c h w -> n h w c')
        color = color[scatter_mask]
        color = color.astype(np.float32)
        # Ensure colors are in [0, 1] range for PyVista
        if len(color) > 0 and color.max() > 1.0:
            color = color / 255.0
        elif len(color) == 0:
            # Fallback empty color array if needed
            color = np.zeros((0, 3), dtype=np.float32)

        def _normalise_to_rgb_range(x, mask):
            x_min, x_max = x.min(), x.max()
            x[~mask.astype(bool)] = 0
            x = (x - x_min) / (x_max - x_min) 
            x[~mask.astype(bool)] = 1
            return x

        x = batch['smpl_T_joints'].reshape(-1, 3)
        
        def _get_view_params(x):
            """Get view parameters with y-axis up, elev=10, azim=20 degrees"""
            max_range = np.array([
                x[:, 0].max() - x[:, 0].min(),
                x[:, 1].max() - x[:, 1].min(),
                x[:, 2].max() - x[:, 2].min()
            ]).max() / 2.0 + 0.1
            mid_x = (x[:, 0].max() + x[:, 0].min()) * 0.5
            mid_y = (x[:, 1].max() + x[:, 1].min()) * 0.5
            mid_z = (x[:, 2].max() + x[:, 2].min()) * 0.5
            center = [mid_x, mid_y, mid_z]
            
            # y-axis up coordinate system: x=right, y=up, z=forward/back
            # elev=10: elevation angle from horizontal (x-z plane)
            # azim=20: rotation around y-axis
            elev, azim = 10, 20
            elev_rad = np.radians(elev)
            azim_rad = np.radians(azim)
            distance = max_range * 3.5  # Increased from 2.5 to move camera further away
            
            # Calculate camera position (from center, looking towards center)
            # Direction from center to camera (spherical coordinates with y-up)
            # x = cos(elev) * sin(azim)
            # y = sin(elev)
            # z = cos(elev) * cos(azim)
            camera_offset = np.array([
                distance * np.cos(elev_rad) * np.sin(azim_rad),
                distance * np.sin(elev_rad),
                distance * np.cos(elev_rad) * np.cos(azim_rad)
            ])
            camera_pos = center + camera_offset
            
            # Front vector points from camera to center (opposite of camera offset)
            front_vec = -camera_offset / np.linalg.norm(camera_offset)
            
            return center, camera_pos, max_range, front_vec
        
        def _render_pointcloud_pyvista(verts, colors=None, center=None, camera_pos=None, max_range=None, front_vec=None):
            """Render point cloud to image using PyVista with y-axis up, elev=10, azim=20"""
            if len(verts) == 0:
                # Return blank image
                img = np.zeros((400, 400, 3), dtype=np.uint8)
                return img
            
            # Create point cloud
            pcd = pv.PolyData(verts)
            
            # Create plotter with explicit offscreen mode
            plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
            
            # PyVista expects colors as uint8
            if colors is not None:
                if colors.max() <= 1.0:
                    colors_uint8 = (colors * 255).astype(np.uint8)
                else:
                    colors_uint8 = colors.astype(np.uint8)
                pcd['colors'] = colors_uint8
                plotter.add_mesh(pcd, point_size=point_size, scalars='colors', 
                                rgb=True, opacity=gt_alpha)
            else:
                plotter.add_mesh(pcd, point_size=point_size, color='blue', opacity=gt_alpha)
            
            # Set camera with y-axis up, elev=10, azim=20
            # Don't use reset_camera() to avoid auto-fitting to individual point clouds
            # This ensures consistent scaling across all renders
            if center is not None and camera_pos is not None:
                plotter.camera.position = camera_pos
                plotter.camera.focal_point = center
                plotter.camera.up = [0, 1, 0]  # y-axis up
                # Set camera distance based on max_range to ensure consistent zoom
                plotter.camera.view_angle = 30.0  # Fixed field of view angle
            elif center is not None:
                plotter.camera.focal_point = center
                plotter.camera.up = [0, 1, 0]  # y-axis up
                plotter.camera.view_angle = 30.0  # Fixed field of view angle
            
            # Render to numpy array
            img = plotter.screenshot(None, return_img=True)
            plotter.close()
            
            return img

        # ---------------------- Canonical stage ----------------------
        if 'imgs' in batch:
            images = rearrange(batch['imgs'][0], 'n c h w -> n h w c')
            images = images.astype(np.float32)

        if 'vc_init' in predictions: # at inference time, we don't have smpl_masks 
            vc_init = _normalise_to_rgb_range(predictions['vc_init'], image_masks_N)
        
        if 'vc_maps' in batch:
            vc_maps = _normalise_to_rgb_range(batch['vc_maps'], smpl_masks)
        elif 'vc_smpl_maps' in batch:
            vc_maps = _normalise_to_rgb_range(batch['vc_smpl_maps'], smpl_masks)

        # ---------------------- Blendshape stage ----------------------
        # Collect all point clouds to compute a combined bounding box for consistent scaling
        all_verts_for_bounds = [x.flatten().reshape(-1, 3)]  # Start with smpl_T_joints
        
        # Add V^c canonical point clouds
        if 'template_mesh_verts' in batch:
            all_verts_for_bounds.append(batch['template_mesh_verts'][0])
        if "vc_init" in predictions:
            vc_init_verts = predictions['vc_init'][0][scatter_mask]
            if len(vc_init_verts) > 0:
                all_verts_for_bounds.append(vc_init_verts)
        if "vc" in predictions:
            vc_verts = predictions['vc'][0]  # k n h w 3
            for k in range(K):
                vc_k_verts = vc_verts[k, scatter_mask]
                if len(vc_k_verts) > 0:
                    all_verts_for_bounds.append(vc_k_verts)
        
        # Add V^p posed point clouds
        if 'vp' in batch:
            vp_list = batch['vp']
            for k in range(K):
                if len(vp_list[k]) > 0:
                    all_verts_for_bounds.append(vp_list[k])
        if "vp_init" in predictions:
            vp_init = predictions['vp_init'][0]  # k n h w 3
            for k in range(K):
                vp_init_k_verts = vp_init[k, scatter_mask]
                if len(vp_init_k_verts) > 0:
                    all_verts_for_bounds.append(vp_init_k_verts)
        if "vp" in predictions:
            vp_verts = predictions['vp'][0]  # k n h w 3
            for k in range(K):
                vp_k_verts = vp_verts[k, scatter_mask]
                if len(vp_k_verts) > 0:
                    all_verts_for_bounds.append(vp_k_verts)
        
        # Compute combined bounding box
        if len(all_verts_for_bounds) > 0:
            all_verts_combined = np.vstack(all_verts_for_bounds)
            center, camera_pos, max_range, front_vec = _get_view_params(all_verts_combined)
        else:
            center, camera_pos, max_range, front_vec = _get_view_params(x)
        
        num_rows = 7
            
        fig = plt.figure(figsize=(subfig_size * K, subfig_size * num_rows))
        r = 0

        # input images
        if 'imgs' in batch:
            for k in range(K):
                ax = fig.add_subplot(num_rows, K, r*K+k+1)
                ax.imshow(images[k])
                ax.set_title(f'Input Image {k}')
                ax.axis('off')
            r += 1

        # Ground truth vp point clouds
        if 'vp' in batch:
            vp_list = batch['vp']
            for k in range(K):
                vp = vp_list[k]
                if len(vp) > 0:
                    img = _render_pointcloud_pyvista(vp, colors=None, center=center, camera_pos=camera_pos, max_range=max_range, front_vec=front_vec)
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.imshow(img)
                    ax.set_title(f'gt vp $V^{k+1}$')
                    ax.axis('off')
                else:
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.text(0.5, 0.5, 'No points', ha='center', va='center')
                    ax.set_title(f'gt vp $V^{k+1}$')
            r += 1

        # predicted initial V_n^c (2D images)
        if 'vc_maps' in batch or 'vc_smpl_maps' in batch:
            for n in range(N):
                ax = fig.add_subplot(num_rows, K, r*K+n+1)
                ax.imshow(vc_maps[0, n])
                ax.set_title(f'gt SMPL $V_{n+1}^c$')
            # Ground truth canonical scatter - place in last subplot of this row
            if 'template_mesh_verts' in batch:
                vc_gt = batch['template_mesh_verts'][0]
                if len(vc_gt) > 0:
                    img = _render_pointcloud_pyvista(vc_gt, colors=None, center=center, camera_pos=camera_pos, max_range=max_range, front_vec=front_vec)
                    ax = fig.add_subplot(num_rows, K, r*K+K)  # Last subplot of this row (K-1+1 = K)
                    ax.imshow(img)
                    ax.set_title(f'gt $V^c$')
                    ax.axis('off')
            r += 1

        if 'vc_init' in predictions:
            for n in range(N):
                ax = fig.add_subplot(num_rows, K, r*K+n+1)
                ax.imshow(vc_init[0, n])
                ax.set_title(f'pred init $V_{n+1}^c$')
            # Predicted initial V^c - place in last subplot of this row
            vc_init_scatter = predictions['vc_init'][0] # n h w 3 
            v = vc_init_scatter[scatter_mask]
            if len(v) > 0:
                img = _render_pointcloud_pyvista(v, colors=color, center=center, camera_pos=camera_pos, max_range=max_range, front_vec=front_vec)
                ax = fig.add_subplot(num_rows, K, r*K+K)  # Last subplot of this row (K-1+1 = K)
                ax.imshow(img)
                ax.set_title(f'pred init $V^c$')
                ax.axis('off')
            r += 1

        # predicted initial V^k scatter
        if "vp_init" in predictions:
            vp_init = predictions['vp_init'][0] # k n h w 3
            for k in range(K):
                verts = vp_init[k, scatter_mask]
                if len(verts) > 0:
                    img = _render_pointcloud_pyvista(verts, colors=color, center=center, camera_pos=camera_pos, max_range=max_range, front_vec=front_vec)
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.imshow(img)
                    ax.set_title(f'pred init $V^{k+1}$')
                    ax.axis('off')
                else:
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.text(0.5, 0.5, 'No points', ha='center', va='center')
                    ax.set_title(f'pred init $V^{k+1}$')
            r += 1

        if "vc" in predictions:
            vc = predictions['vc'][0] # k n h w 3
            for k in range(K):
                verts = vc[k, scatter_mask]
                if len(verts) > 0:
                    img = _render_pointcloud_pyvista(verts, colors=color, center=center, camera_pos=camera_pos, max_range=max_range, front_vec=front_vec)
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.imshow(img)
                    ax.set_title(f'pred $V^{{c,{k+1}}}$')
                    ax.axis('off')
                else:
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.text(0.5, 0.5, 'No points', ha='center', va='center')
                    ax.set_title(f'pred $V^{{c,{k+1}}}$')
            r += 1

        if "vp" in predictions:
            vp = predictions['vp'][0] # k n h w 3
            for k in range(K):
                verts = vp[k, scatter_mask]
                if len(verts) > 0:
                    img = _render_pointcloud_pyvista(verts, colors=color, center=center, camera_pos=camera_pos, max_range=max_range, front_vec=front_vec)
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.imshow(img)
                    ax.set_title(f'pred final $V^{k+1}$')
                    if metrics is not None:
                        ax.set_title(f'pred final $V^{k+1}$')
                    ax.axis('off')
                else:
                    ax = fig.add_subplot(num_rows, K, r*K+k+1)
                    ax.text(0.5, 0.5, 'No points', ha='center', va='center')
                    ax.set_title(f'pred final $V^{k+1}$')

        plt.tight_layout(pad = 0.1)
        plt.savefig(os.path.join(self.save_dir, self._get_filename(f'_{dataset_name}')), dpi=200)
        plt.close()


        
        save=True

        def remove_outliers(points, k=20, threshold=5.0, return_mask=False):
            """Remove statistical outliers using k-nearest neighbors statistics.
            
            Args:
                points: numpy array of shape (N, 3)
                k: number of nearest neighbors to consider
                threshold: threshold multiplier for outlier detection
                return_mask: if True, return (filtered_points, mask) instead of just filtered_points
            
            Returns:
                filtered_points if return_mask=False, else (filtered_points, mask) where mask is boolean array
            """
            if len(points) < k + 1:
                if return_mask:
                    return points, np.ones(len(points), dtype=bool)
                return points
            
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=k+1)
            mean_dist = np.mean(distances[:, 1:], axis=1)
            
            threshold_value = np.mean(mean_dist) + threshold * np.std(mean_dist)
            mask = mean_dist < threshold_value
            
            filtered_points = points[mask]
            if return_mask:
                return filtered_points, mask
            return filtered_points

        if save:
            output_save_dir = os.path.join(self.save_dir, 'point_clouds', f'{self.counter:06d}')
            output_for_vis_save_dir = os.path.join(self.save_dir, 'novel_pose_point_clouds')
            os.makedirs(output_save_dir, exist_ok=True)
            os.makedirs(output_for_vis_save_dir, exist_ok=True)
            
            if 'scan_mesh_verts_centered' in batch:
                gt_vp = batch['scan_mesh_verts_centered']  # list of K meshes
                gt_vp_faces = batch['scan_mesh_faces'][0]  # list of K face arrays
            elif 'scan_verts' in batch:
                gt_vp = batch['scan_verts'][0]  # list of K meshes
                gt_vp_faces = batch['scan_faces'][0]  # list of K face arrays

            pred_vp = predictions['vp'][0] # k n h w 3
            pred_vp_init = predictions['vp_init'][0] # k n h w 3
            
            # Convert colors to uint8 [0, 255] range for trimesh
            color_uint8 = (color * 255.0).astype(np.uint8)[:, :3]
            
            for k in range(K):
                # # Save GT mesh
                # gt_points = gt_vp[k].cpu().detach().numpy()
                # gt_faces = gt_vp_faces[k].cpu().detach().numpy()
                # trimesh.Trimesh(
                #     vertices=gt_points,
                #     faces=gt_faces
                # ).export(os.path.join(output_save_dir, f'gt_vp_{k:03d}.ply'))
                
                # # Save predicted vp point cloud (with colors)
                # pred_points = pred_vp[k, scatter_mask]
                # pred_points, outlier_mask = remove_outliers(pred_points, return_mask=True)
                # # Filter colors to match the points after outlier removal
                # color_filtered = color_uint8[outlier_mask]
                # trimesh.PointCloud(
                #     vertices=pred_points,
                #     colors=color_filtered
                # ).export(os.path.join(output_save_dir, f'pred_vp_{k:03d}.ply'))
                
                # # Save predicted vp_init point cloud (with colors)
                # pred_init_points = pred_vp_init[k, scatter_mask]
                # pred_init_points, outlier_mask_init = remove_outliers(pred_init_points, return_mask=True)
                # # Filter colors to match the init points after outlier removal
                # color_init_filtered = color_uint8[outlier_mask_init]
                # trimesh.PointCloud(
                #     vertices=pred_init_points,
                #     colors=color_init_filtered
                # ).export(os.path.join(output_save_dir, f'pred_vp_init_{k:03d}.ply'))

                if k == K - 1:  # Last frame (K-1 since k is 0-indexed)
                    # Save GT mesh for visualization
                    gt_points = gt_vp[k].cpu().detach().numpy()
                    gt_faces = gt_vp_faces[k].cpu().detach().numpy()
                    trimesh.Trimesh(
                        vertices=gt_points,
                        faces=gt_faces
                    ).export(os.path.join(output_for_vis_save_dir, f'gt_vp_{self.counter:03d}.ply'))
                    
                    # Save predicted vp point cloud (with colors)
                    pred_points = pred_vp[k, scatter_mask]
                    # pred_points, outlier_mask = remove_outliers(pred_points, return_mask=True)
                    # Filter colors to match the points after outlier removal
                    # color_filtered = color_uint8[outlier_mask]
                    trimesh.PointCloud(
                        vertices=pred_points,
                        colors=color_uint8#color_filtered
                    ).export(os.path.join(output_for_vis_save_dir, f'pred_vp_{self.counter:03d}.ply'))
                    
                    # Save predicted vp_init point cloud (with colors)
                    pred_init_points = pred_vp_init[k, scatter_mask]
                    # pred_init_points, outlier_mask_init = remove_outliers(pred_init_points, return_mask=True)
                    # Filter colors to match the init points after outlier removal
                    # color_init_filtered = color_uint8[outlier_mask_init]
                    trimesh.PointCloud(
                        vertices=pred_init_points,
                        colors=color_uint8#color_init_filtered
                    ).export(os.path.join(output_for_vis_save_dir, f'pred_vp_init_{self.counter:03d}.ply'))

            self.counter += 1




    def visualise_full(
        self,
        predictions,
        batch
    ):
        subfig_size = 4
        s = 0.05 # scatter point size
        gt_alpha, pred_alpha = 0.5, 0.5

        B, N, H, W, C = predictions['vc_init'].shape
        K = 5

        mask = batch['masks']
        mask_N = mask[:, :N]

        num_rows = 0


        scatter_mask = mask_N[0].astype(bool) # nhw
        if "vc_init_conf" in predictions:
            confidence = predictions['vc_init_conf']
            # Compute threshold from percentage if enabled, otherwise use fixed threshold
            threshold_value = self._get_confidence_threshold_from_percentage(
                confidence[0], mask_N[0]
            )
            confidence = confidence > threshold_value
        else:
            confidence = np.ones_like(predictions['vc_init'])[..., 0].astype(bool)
        scatter_mask = scatter_mask * confidence[0].astype(bool)
        
        # If scatter_mask is empty (all confidence below threshold), fall back to image masks only
        if not np.any(scatter_mask):
            scatter_mask = mask_N[0].astype(bool)
        
        # Color for predicted scatters 
        color = rearrange(batch['imgs'][0, :N], 'n c h w -> n h w c')
        color = color[scatter_mask]
        # color = (color * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
        if len(color) > 0:
            color = color.astype(np.float32)
        else:
            # Fallback empty color array if needed
            color = np.zeros((0, 3), dtype=np.float32)
        


        def _normalise_to_rgb_range(x, mask):
            # x_min, x_max = -1.4, 1.1 
            x_min, x_max = x.min(), x.max()
            # assert x_min <= x.min() and x.max() <= x_max
            x[~mask.astype(bool)] = 0
            x = (x - x_min) / (x_max - x_min) 
            x[~mask.astype(bool)] = 1
            return x
        

        # x = batch['template_mesh_verts'][0]
        x = batch['smpl_T_joints'].reshape(-1, 3)
        def _set_scatter_limits(ax, x):
            max_range = np.array([
                x[:, 0].max() - x[:, 0].min(),
                x[:, 1].max() - x[:, 1].min(),
                x[:, 2].max() - x[:, 2].min()
            ]).max() / 2.0 + 0.1
            mid_x = (x[:, 0].max() + x[:, 0].min()) * 0.5
            mid_y = (x[:, 1].max() + x[:, 1].min()) * 0.5
            mid_z = (x[:, 2].max() + x[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.view_init(elev=10, azim=20, vertical_axis='y')
            ax.set_box_aspect([1, 1, 1])
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5)) 
            ax.zaxis.set_major_locator(plt.MaxNLocator(3))

        # ---------------------- Canonical stage ----------------------
        if 'imgs' in batch:
            images = rearrange(batch['imgs'][0], 'n c h w -> n h w c')
            # images = (images * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
            images = images.astype(np.float32)

        if 'vc_init' in predictions:
            vc_init = _normalise_to_rgb_range(predictions['vc_init'], mask_N)
        
        if 'vc_init_conf' in predictions:
            vc_init_conf = predictions['vc_init_conf']
            vc_init_conf = vc_init_conf * mask_N

        if 'vc_maps' in batch:
            vc_maps = _normalise_to_rgb_range(batch['vc_maps'], batch['smpl_mask'])

        if 'vc_smpl_maps' in batch:
            vc_maps = _normalise_to_rgb_range(batch['vc_smpl_maps'], batch['smpl_mask'])

        if 'smpl_w_maps' in batch:
            smpl_w_maps = np.argmax(batch['smpl_w_maps'], axis=-1)

        if "w" in predictions:
            w = predictions['w']
            w = np.argmax(w, axis=-1)
            w = w * mask_N

        # ---------------------- Blendshape stage ----------------------
        mask = np.repeat(mask_N[:, None, :, :], K, axis=1) # B, K, N, H, W 
    
        if 'vp_init' in predictions:
            vp_init = _normalise_to_rgb_range(predictions['vp_init'], mask)

        if 'vc' in predictions:
            vc = _normalise_to_rgb_range(predictions['vc'], mask)

        if 'vp' in predictions:
            vp = _normalise_to_rgb_range(predictions['vp'], mask)

        if 'dvc' in predictions:
            dvc = predictions['dvc'] # bknhwc
            dvc = np.linalg.norm(dvc, axis=-1)
            dvc = dvc * mask


        num_rows = 8
            
            
        fig = plt.figure(figsize=(subfig_size * K, subfig_size * num_rows))
        r = 0

        # input images
        if 'imgs' in batch:
            for k in range(K):
                ax = fig.add_subplot(num_rows, K, r*K+k+1)
                ax.imshow(images[k])
                ax.set_title(f'Input Image {k}')
                ax.axis('off')
            r += 1

        # scan meshes
        # if "scan_mesh_verts_centered" in batch:
        #     scan_mesh_verts = batch['scan_mesh_verts_centered'][0]
        #     scan_mesh_colors = batch['scan_mesh_colors'][0]
        #     for k in range(K):
        #         verts = scan_mesh_verts[k].cpu().detach().numpy()
        #         colors = (scan_mesh_colors[k].cpu().detach().numpy() / 255.).astype(np.float32)
        #         ax = fig.add_subplot(num_rows, K, r*K+k+1, projection='3d')
        #         ax.scatter(verts[:, 0], 
        #                    verts[:, 1], 
        #                    verts[:, 2], c=colors, s=s, alpha=gt_alpha)
        #         ax.set_title(f'gt scan $V^{k+1}$')
        #         _set_scatter_limits(ax, x)
        #     r += 1

        if 'vp' in batch:
            # vp_ptcld = batch['vp_ptcld']
            # vp_list = vp_ptcld.points_list()
            vp_list = batch['vp']
            for k in range(K):
                vp = vp_list[k]#.cpu().detach().numpy()
                ax = fig.add_subplot(num_rows, K, r*K+k+1, projection='3d')
                ax.scatter(vp[:, 0], 
                           vp[:, 1], 
                           vp[:, 2], s=s, alpha=gt_alpha)
                ax.set_title(f'gt vp $V^{k+1}$')
                _set_scatter_limits(ax, x)
            r += 1


        # predicted initial V_n^c
        if 'vc_maps' in batch or 'vc_smpl_maps' in batch:
            for n in range(N):
                ax = fig.add_subplot(num_rows, K, r*K+n+1)
                ax.imshow(vc_maps[0, n])
                ax.set_title(f'gt SMPL $V_{n}^c$')
            r += 1

        if 'vc_init' in predictions:
            for n in range(N):
                ax = fig.add_subplot(num_rows, K, r*K+n+1)
                ax.imshow(vc_init[0, n])
                ax.set_title(f'$V_c$ init {n}')
            r += 1


        if 'template_mesh_verts' in batch:
            # Additional row for gt canonical scatter and initial predictions
            ax = fig.add_subplot(num_rows, K, r*K+1, projection='3d')
            vc_gt = batch['template_mesh_verts'][0]#.cpu().detach().numpy()
            ax.scatter(vc_gt[:, 0], 
                    vc_gt[:, 1], 
                    vc_gt[:, 2], c='blue', s=s, alpha=gt_alpha, label=f'$V^c$')
            _set_scatter_limits(ax, x)
            ax.set_title(f'gt $V^c$')


        if "vc_init" in predictions:
            ax = fig.add_subplot(num_rows, K, r*K+2, projection='3d')
            vc_init = predictions['vc_init'][0] # n h w 3 
            
            v = vc_init[scatter_mask]
            ax.scatter(v[:, 0], 
                       v[:, 1], 
                       v[:, 2], c=color, s=s, alpha=gt_alpha, label=f'$V^c$ init')
            _set_scatter_limits(ax, x)
            ax.set_title(f'pred init $V^c$')
            r += 1


        # predicted initial V^k scatter
        if "vp_init" in predictions:
            vp_init = predictions['vp_init'][0] # k n h w 3

            for k in range(K):
                verts = vp_init[k, scatter_mask]
                ax = fig.add_subplot(num_rows, K, r*K+k+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                ax.set_title(f'pred init $V^{k+1}$')
                _set_scatter_limits(ax, x)
            r += 1

        

        if "vc" in predictions:
            vc = predictions['vc'][0] # k n h w 3

            for k in range(K):
                verts = vc[k, scatter_mask]
                ax = fig.add_subplot(num_rows, K, r*K+k+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                _set_scatter_limits(ax, x)
                ax.set_title(f'pred $V^{{c,{k+1}}}$')
            r += 1

        if "vp" in predictions:
            vp = predictions['vp'][0] # k n h w 3

            for k in range(K):
                verts = vp[k, scatter_mask]
                ax = fig.add_subplot(num_rows, K, r*K+k+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                _set_scatter_limits(ax, x)
                ax.set_title(f'pred final $V^{k+1}$')


        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, self._get_filename()), dpi=200)

        plt.close()
            



    def visualise_vp_vc(
        self,
        predictions,
        batch
    ):

        """
        Visualise per-frame posed vertices and joined canonical points

        Row 1: ground truth vp, ground truth first frame v_cano
        Row 2: pred initial vp, red initial vc (w/o pbs)
        Row 3: pred vp w/ pbs, pred vc

        Args:
            vp: (B, N, 6890, 3) CAPE scan vertices
            vp_init_pred: (B, K, N, H, W, 3): w/o pose blendshapes
            vp_pred: (B, K, N, H, W, 3)

            vc: (B, N, H, W, 3)
            vc_init_pred: (B, N, H, W, 3): w/o pose blendshapes
            vc_pred: (B, K, N, H, W, 3)

            bg_mask: (B, N, H, W) background mask
            conf_mask: (B, N, H, W) canonical points confidence mask
        """
        gt_alpha = 0.5
        pred_alpha = 0.5
        s = 0.1
        
        num_rows, num_cols = 4, 5
        sub_fig_size = 4 

        # Get N from predictions shape
        B, N, H, W, C = predictions['vc_init'].shape

        mask = batch['masks'][0].astype(np.bool) # nhw
        if "vc_init_conf" in predictions:
            confidence = predictions['vc_init_conf']
            # Compute threshold from percentage if enabled, otherwise use fixed threshold
            threshold_value = self._get_confidence_threshold_from_percentage(
                confidence[0], batch['masks'][0]
            )
            confidence = confidence > threshold_value
        else:
            confidence = np.ones_like(predictions['vc_init'])[..., 0].astype(np.bool)
        mask = mask * confidence[0].astype(np.bool)


        color = rearrange(batch['imgs'][0], 'n c h w -> n h w c')
        color = color[mask]
        color = (color * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
        color = color.astype(np.float32)
        


        fig = plt.figure(figsize=(sub_fig_size * num_cols, sub_fig_size * num_rows))

        ax = fig.add_subplot(num_rows, num_cols, 5, projection='3d')
        vc_gt = batch['template_mesh_verts'][0]#.cpu().detach().numpy()
        ax.scatter(vc_gt[:, 0], 
                   vc_gt[:, 1], 
                   vc_gt[:, 2], c='blue', s=s, alpha=gt_alpha, label=f'$V^c$')
        ax.set_title(f'gt $V^c$')

        if "vc_init" in predictions:
            ax = fig.add_subplot(num_rows, num_cols, num_cols+5, projection='3d')
            vc_init = predictions['vc_init'][0] # n h w 3 
            
            verts = vc_init[mask]
            ax.scatter(verts[:, 0], 
                       verts[:, 1], 
                       verts[:, 2], c=color, s=s, alpha=gt_alpha, label=f'$V^c$ init')
            ax.set_title(f'pred init $V^c$')
            
        if "scan_mesh_verts_centered" in batch:
            scan_mesh_verts = batch['scan_mesh_verts_centered'][0]
            scan_mesh_colors = batch['scan_mesh_colors'][0]
            N_vis = min(N, self.num_frames_pp)
            for i in range(N_vis):
                verts = scan_mesh_verts[i].cpu().detach().numpy()
                colors = (scan_mesh_colors[i].cpu().detach().numpy() / 255.).astype(np.float32)
                ax = fig.add_subplot(num_rows, num_cols, i+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=colors, s=s, alpha=gt_alpha)
                ax.set_title(f'gt scan $V^{i+1}$')

        if "vp_init" in predictions:
            vp_init = predictions['vp_init'][0] # k n h w 3
            J_init = predictions['J_init'][0]
            N_vis = min(N, self.num_frames_pp)

            for i in range(N_vis):
                verts = vp_init[i, mask]
                ax = fig.add_subplot(num_rows, num_cols, num_cols+i+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                # ax.scatter(J_init[i, :, 0], 
                #            J_init[i, :, 1], 
                #            J_init[i, :, 2], c='green', s=1., alpha=gt_alpha')
                ax.set_title(f'pred init $V^{i+1}$')


        if "vc" in predictions:
            vc = predictions['vc'][0] # k n h w 3
            N_vis = min(N, self.num_frames_pp)

            for i in range(N_vis):
                verts = vc[i, mask]
                ax = fig.add_subplot(num_rows, num_cols, 2*num_cols+i+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                ax.set_title(f'pred $V^{{c,{i+1}}}$')
                
        if "vp" in predictions:
            vp = predictions['vp'][0] # k n h w 3
            N_vis = min(N, self.num_frames_pp)

            for i in range(N_vis):
                verts = vp[i, mask]
                ax = fig.add_subplot(num_rows, num_cols, 3*num_cols+i+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                ax.set_title(f'pred final $V^{i+1}$')


        x = batch['template_mesh_verts'][0]#.cpu().detach().numpy()
        max_range = np.array([
            x[:, 0].max() - x[:, 0].min(),
            x[:, 1].max() - x[:, 1].min(),
            x[:, 2].max() - x[:, 2].min()
        ]).max() / 2.0
        mid_x = (x[:, 0].max() + x[:, 0].min()) * 0.5
        mid_y = (x[:, 1].max() + x[:, 1].min()) * 0.5
        mid_z = (x[:, 2].max() + x[:, 2].min()) * 0.5

        for ax in fig.axes:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.view_init(elev=10, azim=20, vertical_axis='y')
            ax.set_box_aspect([1, 1, 1])
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5)) 
            ax.zaxis.set_major_locator(plt.MaxNLocator(3))
        # self._no_annotations(fig)

        plt.tight_layout(h_pad=4)
        plt.savefig(os.path.join(self.save_dir, self._get_filename('_vp_vc')), dpi=300)

        plt.close()

        # fig = plt.figure(figsize=(8, 4))

        # ax = fig.add_subplot(1, 2, 1)
        # vc_gt = batch['template_mesh_verts'][0]#.cpu().detach().numpy()
        # ax.scatter(vc_gt[:, 0], 
        #            vc_gt[:, 1], c='blue', s=s, alpha=gt_alpha, label=f'$vc_{0}$')
        # ax.set_title(f'gt $V_c$ {0}')

        # ax = fig.add_subplot(1, 2, 2)
        # vc_init = predictions['vc_init'][0] # n h w 3 
        
        # verts = vc_init[mask]
        # ax.scatter(verts[:, 0], 
        #             verts[:, 1], c=color, s=s, alpha=gt_alpha, label=f'$vc_init_{0}$')
        # ax.set_title(f'pred $V_c$ init')

        # # Set equal plot range based on vc_gt
        # max_range = np.array([
        #     vc_gt[:, 0].max() - vc_gt[:, 0].min(),
        #     vc_gt[:, 1].max() - vc_gt[:, 1].min()
        # ]).max() / 2.0
        # mid_x = (vc_gt[:, 0].max() + vc_gt[:, 0].min()) * 0.5
        # mid_y = (vc_gt[:, 1].max() + vc_gt[:, 1].min()) * 0.5

        # for ax in fig.axes:
        #     ax.set_xlim(mid_x - max_range - 0.1, mid_x + max_range + 0.1)
        #     ax.set_ylim(mid_y - max_range - 0.1, mid_y + max_range + 0.1)
        #     ax.set_aspect('equal')

        # plt.tight_layout()
        # plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_test.png'), dpi=300)

        # import ipdb; ipdb.set_trace()


        # plt.close()






    # def visualise_scenepic(
    #     self,
    #     predictions,
    #     batch
    # ):
    #     viridis = plt.colormaps.get_cmap('viridis')
        
    #     B, N = vp.shape[:2]
    #     B, N = min(B, 1), min(N, 4)

    #     scene = sp.Scene()


    #     # ----------------------- vc pred -----------------------
    #     positions = []
    #     colors = []
    #     for n in range(N):
    #         vc_plot = vc_pred[0, n, masks[0, n].astype(np.bool).flatten()].reshape(-1, 3)
    #         vc_plot[..., 0] += 2.0
    #         positions.append(vc_plot)
    #         colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())

    #     positions = np.concatenate(positions, axis=0)
    #     colors = np.concatenate(colors, axis=0)
    #     colors_normalized = colors / colors.max()
    #     colors_rgb = viridis(colors_normalized)[:, :3] 

    #     mesh_vc_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vc_pred")
    #     mesh_vc_pred.add_sphere() 
    #     mesh_vc_pred.apply_transform(sp.Transforms.Scale(0.005)) 
    #     mesh_vc_pred.enable_instancing(positions = positions, colors = colors_rgb) 


    #     # ----------------------- vc gt -----------------------
    #     positions = []
    #     for n in range(N):
    #         vc_gt = vc[0, n, masks[0, n].astype(np.bool)].reshape(-1, 3)
    #         vc_gt[..., 0] += 2.0
    #         positions.append(vc_gt)
    #     positions = np.concatenate(positions, axis=0)

    #     mesh_vc_gt = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vc_gt")
    #     mesh_vc_gt.add_sphere() 
    #     mesh_vc_gt.apply_transform(sp.Transforms.Scale(0.005)) 
    #     mesh_vc_gt.enable_instancing(positions = positions) 



    #     # ----------------------- vp pred -----------------------
    #     positions = []
    #     colors = []

    #     for n in range(N):
    #         vp_plot = vp_pred[0, n, masks[0, n].astype(np.bool).flatten()]
    #         vp_plot[..., 0] += 0.8 * n - 1.6
    #         positions.append(vp_plot)
    #         colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())


    #     positions = np.concatenate(positions, axis=0)
    #     colors = np.concatenate(colors, axis=0)


    #     colors_normalized = colors / colors.max()
    #     colors_rgb = viridis(colors_normalized)[:, :3] 

    #     mesh_vp_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vp_pred")
    #     mesh_vp_pred.add_sphere() 
    #     mesh_vp_pred.apply_transform(sp.Transforms.Scale(0.005)) 
    #     mesh_vp_pred.enable_instancing(positions = positions, colors = colors_rgb) 


    #     # ----------------------- vp gt -----------------------
    #     positions = []
    #     for n in range(N):
    #         vp_gt_plot = vp[0, n, vertex_visibility[0, n].astype(np.bool).flatten()]
    #         vp_gt_plot[..., 0] += 0.8 * n - 1.6
    #         positions.append(vp_gt_plot)
    #     positions = np.concatenate(positions, axis=0)

    #     mesh_vp_gt = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vp_gt")
    #     mesh_vp_gt.add_sphere() 
    #     mesh_vp_gt.apply_transform(sp.Transforms.Scale(0.005)) 
    #     mesh_vp_gt.enable_instancing(positions = positions) 



    #     # ----------------------- vp init pred -----------------------
    #     if vp_init_pred is not None:

    #         # vp gt overlay 
    #         positions = []
    #         for n in range(N):
    #             vp_gt_plot = vp[0, n, vertex_visibility[0, n].astype(np.bool).flatten()]
    #             vp_gt_plot[..., 0] += 0.8 * n - 1.6
    #             vp_gt_plot[..., 1] -= 2.0
    #             positions.append(vp_gt_plot)
    #         positions = np.concatenate(positions, axis=0)

    #         mesh_vp_gt_row_2 = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vp_gt_row_2")
    #         mesh_vp_gt_row_2.add_sphere() 
    #         mesh_vp_gt_row_2.apply_transform(sp.Transforms.Scale(0.005)) 
    #         mesh_vp_gt_row_2.enable_instancing(positions = positions) 

    #         # vp init pred 
    #         positions = []
    #         colors = []
    #         for n in range(N):
    #             vp_init_pred_plot = vp_init_pred[0, n, masks[0, n].astype(np.bool).flatten()]
    #             vp_init_pred_plot[..., 0] += 0.8 * n - 1.6
    #             vp_init_pred_plot[..., 1] -= 2.0
    #             positions.append(vp_init_pred_plot)
    #             colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())
    #         positions = np.concatenate(positions, axis=0)
    #         colors = np.concatenate(colors, axis=0)
    #         colors_normalized = colors / colors.max()
    #         colors_rgb = viridis(colors_normalized)[:, :3] 

    #         mesh_vp_init_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vp_init_pred")
    #         mesh_vp_init_pred.add_sphere() 
    #         mesh_vp_init_pred.apply_transform(sp.Transforms.Scale(0.005)) 
    #         mesh_vp_init_pred.enable_instancing(positions = positions, colors = colors_rgb) 

    #     if vc_init_pred is not None:
    #         # vc gt overlay 
    #         positions = []
    #         for n in range(N):
    #             vc_gt = vc[0, n, masks[0, n].astype(np.bool)].reshape(-1, 3)
    #             vc_gt[..., 0] += 2.0
    #             positions.append(vc_gt)
    #         positions = np.concatenate(positions, axis=0)

    #         mesh_vc_gt_row_2 = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vc_gt_row_2")
    #         mesh_vc_gt_row_2.add_sphere() 
    #         mesh_vc_gt_row_2.apply_transform(sp.Transforms.Scale(0.005)) 
    #         mesh_vc_gt_row_2.enable_instancing(positions = positions) 



    #         positions = []
    #         colors = []
    #         for n in range(N):
    #             vc_init_pred_plot = vc_init_pred[0, n, masks[0, n].astype(np.bool).flatten()]
    #             vc_init_pred_plot[..., 0] += 2.0
    #             vc_init_pred_plot[..., 1] -= 2.0
    #             positions.append(vc_init_pred_plot)
    #             colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())
    #         positions = np.concatenate(positions, axis=0)
    #         colors = np.concatenate(colors, axis=0)
    #         colors_normalized = colors / colors.max()
    #         colors_rgb = viridis(colors_normalized)[:, :3] 

    #         mesh_vc_init_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vc_init_pred")
    #         mesh_vc_init_pred.add_sphere() 
    #         mesh_vc_init_pred.apply_transform(sp.Transforms.Scale(0.005)) 
    #         mesh_vc_init_pred.enable_instancing(positions = positions, colors = colors_rgb) 


    #     # ----------------------- canvas -----------------------
    #     golden_ratio = (1 + np.sqrt(5)) / 2
    #     canvas = scene.create_canvas_3d(width = 1600, height = 1600 / golden_ratio, shading=sp.Shading(bg_color=sp.Colors.White))
    #     frame = canvas.create_frame()

    #     frame.add_mesh(mesh_vp_pred)
    #     frame.add_mesh(mesh_vp_gt)

    #     frame.add_mesh(mesh_vc_pred)
    #     frame.add_mesh(mesh_vc_gt)

    #     if vp_init_pred is not None:
    #         frame.add_mesh(mesh_vp_init_pred)
    #         frame.add_mesh(mesh_vp_gt_row_2)
    #     if vc_init_pred is not None:
    #         frame.add_mesh(mesh_vc_init_pred)
    #         frame.add_mesh(mesh_vc_gt_row_2)

    #     path = os.path.join(self.save_dir, f'{self.global_step:06d}_sp.html')

    #     scene.save_as_html(path)
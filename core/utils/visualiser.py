import torch 
import os 
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl 
import matplotlib.colors
# import scenepic as sp 
from einops import rearrange
from collections import defaultdict

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Visualiser(pl.LightningModule):
    def __init__(self, save_dir, cfg=None, rank=0):
        super().__init__()
        self.save_dir = save_dir
        self.rank = rank
        self.cfg = cfg
        self.threshold = cfg.LOSS.CONFIDENCE_THRESHOLD if cfg is not None else 100

        # self.threshold = 50

    def set_global_rank(self, global_rank):
        self.rank = global_rank


    def visualise(
        self, 
        predictions,
        batch
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
        

        # Convert predictions to numpy if tensor
        for k, v in predictions.items():
            predictions[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items():
            batch[k] = v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v

        # self.visualise_input_normal_imgs(
        #     normal_maps
        # )

        self.visualise_initial_pms(
            predictions,
            batch
        )

        self.visualise_pbs_pms(
            predictions,
            batch
        )

        self.visualise_full(
            predictions,
            batch
        )

        # self.visualise_vp_vc(
        #     predictions,
        #     batch
        # )


        
        

    def visualise_input_images(self, images):
        if self.rank == 0:
            B, N = images.shape[:2]
            B = min(B, 2)
            N = min(N, 4)

            
            images = np.transpose(images, (0, 1, 3, 4, 2))

            # Create a grid of subplots
            fig, axes = plt.subplots(B, N, figsize=(4*N, 4*B))
            
            # Handle single row/column case
            if B == 1:
                axes = axes[None, :]
            if N == 1:
                axes = axes[:, None]

            # Plot each normal image
            for b in range(B):
                for n in range(N):
                    axes[b,n].imshow(images[b,n])
                    # axes[b,n].axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_images.png'))
            plt.close()
      

    def visualise_initial_pms(
        self, 
        predictions,
        batch
    ):
        B, N, H, W, C = predictions['vc_init'].shape
        mask = batch['masks']

        mask_N = mask[:, :N]

        B = 1
        N = min(N, 4)
        sub_fig_size = 4
        num_cols = 4

        num_rows = 0

        if 'imgs' in batch:
            num_rows += 1
            images = rearrange(batch['imgs'][0], 'n c h w -> n h w c')
            images = (images * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
            images = images.astype(np.float32)

        if 'vc_init' in predictions:
            num_rows += 1
            vc_init = predictions['vc_init']
            vc_init[~mask_N.astype(bool)] = 0

            norm_min, norm_max = vc_init.min(), vc_init.max()
            vc_init = (vc_init - norm_min) / (norm_max - norm_min) 
            vc_init[~mask_N.astype(bool)] = 1
        
        if 'vc_init_conf' in predictions:
            num_rows += 1
            vc_init_conf = predictions['vc_init_conf']
            vc_init_conf = vc_init_conf * mask_N 

        if 'vc_maps' in batch:
            num_rows += 1
            vc_maps = batch['vc_maps']
            smpl_mask = batch['smpl_mask']
            vc_maps[~smpl_mask.astype(bool)] = 0
            vc_maps = (vc_maps - vc_maps.min()) / (vc_maps.max() - vc_maps.min())
            vc_maps[~smpl_mask.astype(bool)] = 1

        if 'smpl_w_maps' in batch:
            num_rows += 1
            smpl_w_maps = np.argmax(batch['smpl_w_maps'], axis=-1)

        if "w" in predictions:
            num_rows += 1
            w = predictions['w']
            w = np.argmax(w, axis=-1)
            w = w * mask_N 

    
        fig = plt.figure(figsize=(num_cols*sub_fig_size, num_rows*sub_fig_size))
        for n in range(num_cols):
            row = 0

            if 'imgs' in batch:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(images[n])
                plt.title(f'Image {n}')
                row += 1

            if 'vc_maps' in batch:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(vc_maps[0, n])
                plt.title(f'$V_c$ maps {n}')
                row += 1
                
            if 'vc_init' in predictions:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(vc_init[0, n])
                plt.title(f'$V_c$ init {n}')
                row += 1

            if 'vc_init_conf' in predictions:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(vc_init_conf[0, n])
                plt.title(f'$V_c$ conf {n}')
                if n == 3:
                    plt.colorbar()
                row += 1

            if 'smpl_w_maps' in batch:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(smpl_w_maps[0, n])
                plt.title(f'Smpl $w$ maps {n}')
                row += 1
            
            if "w" in predictions:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(w[0, n])
                plt.title(f'Pred $w$ maps {n}')
                row += 1

        # for ax in fig.axes:
            # ax.set_xticks([])
            # ax.set_yticks([])


        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_pms.png'))
        plt.close()



    def visualise_pbs_pms(
        self,
        predictions,
        batch
    ):
        
        B, N, H, W, C = predictions['vc_init'].shape
        K = 5 
        mask = np.repeat(batch['masks'][:, :N][:, None], K, axis=1) # B, K, N, H, W

        B = 1
        N = min(N, 4)
        sub_fig_size = 4
        num_cols = 4

        num_rows = 0

        if 'vp_init' in predictions:
            num_rows += 4
            vp_init = predictions['vp_init'] # bknhwc
            vp_init[~mask.astype(bool)] = 0
            norm_min, norm_max = vp_init.min(), vp_init.max()
            vp_init = (vp_init - norm_min) / (norm_max - norm_min) 
            vp_init[~mask.astype(bool)] = 1


        if 'vc' in predictions:
            num_rows += 4
            vc = predictions['vc'] # bknhwc
            vc[~mask.astype(bool)] = 0
            norm_min, norm_max = vc.min(), vc.max()
            vc = (vc - norm_min) / (norm_max - norm_min) 
            vc[~mask.astype(bool)] = 1

        if 'vp' in predictions:
            num_rows += 4
            vp = predictions['vp'] # bknhwc
            vp[~mask.astype(bool)] = 0
            norm_min, norm_max = vp.min(), vp.max()
            vp = (vp - norm_min) / (norm_max - norm_min) 
            vp[~mask.astype(bool)] = 1

        if 'dvc' in predictions:
            num_rows += 4
            dvc = predictions['dvc'] # bknhwc
            dvc = np.linalg.norm(dvc, axis=-1)
            dvc = dvc * mask


        fig = plt.figure(figsize=(num_cols*sub_fig_size, num_rows*sub_fig_size))
        row = 0
        for k in range(num_cols):
            if 'vp_init' in predictions:
                for n in range(4):
                    plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                    plt.imshow(vp_init[0, k, n])
                    plt.title(f'Pred init $V_{n+1}^{k+1}$')
                row += 1

            if 'dvc' in predictions:
                for n in range(4):
                    ax = plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                    im = ax.imshow(dvc[0, k, n])
                    plt.title(f'Pred $\\Delta V_{n+1}^{{c,{k+1}}}$')
                    plt.colorbar(im, ax=ax)
                row += 1

            if 'vc' in predictions:
                for n in range(4):
                    plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                    plt.imshow(vc[0, k, n])
                    plt.title(f'Pred $V_{n+1}^{{c,{k+1}}}$')
                row += 1

            if 'vp' in predictions:
                for n in range(4):
                    plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                    plt.imshow(vp[0, k, n])
                    plt.title(f'Pred final $V_{n+1}^{k+1}$')
                row += 1
            


        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_pbs_pms.png'))
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


        mask = batch['masks'][0].astype(np.bool) # nhw
        if "vc_init_conf" in predictions:
            confidence = predictions['vc_init_conf']
            confidence = confidence > self.threshold
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
            for i in range(4):
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

            for i in range(4):
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

            for i in range(4):
                verts = vc[i, mask]
                ax = fig.add_subplot(num_rows, num_cols, 2*num_cols+i+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                ax.set_title(f'pred $V^{{c,{i+1}}}$')
                
        if "vp" in predictions:
            vp = predictions['vp'][0] # k n h w 3

            for i in range(4):
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
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_vp_vc.png'), dpi=300)

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




    def visualise_full(
        self,
        predictions,
        batch
    ):
        subfig_size = 4
        s = 0.1 # scatter point size
        gt_alpha, pred_alpha = 0.5, 0.5

        B, N, H, W, C = predictions['vc_init'].shape
        K = 5

        mask = batch['masks']
        mask_N = mask[:, :N]

        num_rows = 0


        scatter_mask = mask_N[0].astype(bool) # nhw
        if "vc_init_conf" in predictions:
            confidence = predictions['vc_init_conf']
            confidence = confidence > self.threshold
        else:
            confidence = np.ones_like(predictions['vc_init'])[..., 0].astype(bool)
        scatter_mask = scatter_mask * confidence[0].astype(bool)
        # Color for predicted scatters 
        color = rearrange(batch['imgs'][0, :N], 'n c h w -> n h w c')
        color = color[scatter_mask]
        color = (color * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
        color = color.astype(np.float32)
        


        def _normalise_to_rgb_range(x, mask):
            # x_min, x_max = -1.4, 1.1 
            x_min, x_max = x.min(), x.max()
            # assert x_min <= x.min() and x.max() <= x_max
            x[~mask.astype(bool)] = 0
            x = (x - x_min) / (x_max - x_min) 
            x[~mask.astype(bool)] = 1
            return x
        

        x = batch['template_mesh_verts'][0]
        def _set_scatter_limits(ax, x):
            max_range = np.array([
                x[:, 0].max() - x[:, 0].min(),
                x[:, 1].max() - x[:, 1].min(),
                x[:, 2].max() - x[:, 2].min()
            ]).max() / 2.0
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
            images = (images * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
            images = images.astype(np.float32)

        if 'vc_init' in predictions:
            vc_init = _normalise_to_rgb_range(predictions['vc_init'], mask_N)
        
        if 'vc_init_conf' in predictions:
            vc_init_conf = predictions['vc_init_conf']
            vc_init_conf = vc_init_conf * mask_N

        if 'vc_maps' in batch:
            vc_maps = _normalise_to_rgb_range(batch['vc_maps'], batch['smpl_mask'])

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
            
            
        fig = plt.figure(figsize=(subfig_size * N, subfig_size * num_rows))
        r = 0

        # input images
        if 'imgs' in batch:
            for n in range(N):
                ax = fig.add_subplot(num_rows, N, r*N+n+1)
                ax.imshow(images[n])
                ax.set_title(f'Input Image {n}')
                ax.axis('off')
            r += 1

        # scan meshes
        if "scan_mesh_verts_centered" in batch:
            scan_mesh_verts = batch['scan_mesh_verts_centered'][0]
            scan_mesh_colors = batch['scan_mesh_colors'][0]
            for n in range(N):
                verts = scan_mesh_verts[n].cpu().detach().numpy()
                colors = (scan_mesh_colors[n].cpu().detach().numpy() / 255.).astype(np.float32)
                ax = fig.add_subplot(num_rows, N, r*N+n+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=colors, s=s, alpha=gt_alpha)
                ax.set_title(f'gt scan $V^{n+1}$')
                _set_scatter_limits(ax, x)
            r += 1

        # predicted initial V_n^c
        if 'vc_maps' in batch:
            for n in range(N):
                ax = fig.add_subplot(num_rows, N, r*N+n+1)
                ax.imshow(vc_maps[0, n])
                ax.set_title(f'gt SMPL $V_{n}^c$')
            r += 1

        if 'vc_init' in predictions:
            for n in range(N):
                ax = fig.add_subplot(num_rows, N, r*N+n+1)
                ax.imshow(vc_init[0, n])
                ax.set_title(f'$V_c$ init {n}')
            r += 1


        # Additional row for gt canonical scatter and initial predictions
        ax = fig.add_subplot(num_rows, N, r*N+1, projection='3d')
        vc_gt = batch['template_mesh_verts'][0]#.cpu().detach().numpy()
        ax.scatter(vc_gt[:, 0], 
                   vc_gt[:, 1], 
                   vc_gt[:, 2], c='blue', s=s, alpha=gt_alpha, label=f'$V^c$')
        _set_scatter_limits(ax, x)
        ax.set_title(f'gt $V^c$')


        if "vc_init" in predictions:
            ax = fig.add_subplot(num_rows, N, r*N+2, projection='3d')
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

            for n in range(N):
                verts = vp_init[n, scatter_mask]
                ax = fig.add_subplot(num_rows, N, r*N+n+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                ax.set_title(f'pred init $V^{n+1}$')
                _set_scatter_limits(ax, x)
            r += 1

        

        if "vc" in predictions:
            vc = predictions['vc'][0] # k n h w 3

            for n in range(N):
                verts = vc[n, scatter_mask]
                ax = fig.add_subplot(num_rows, N, r*N+n+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                _set_scatter_limits(ax, x)
                ax.set_title(f'pred $V^{{c,{n+1}}}$')
            r += 1

        if "vp" in predictions:
            vp = predictions['vp'][0] # k n h w 3

            for n in range(N):
                verts = vp[n, scatter_mask]
                ax = fig.add_subplot(num_rows, N, r*N+n+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha)
                _set_scatter_limits(ax, x)
                ax.set_title(f'pred final $V^{n+1}$')


        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}.png'), dpi=200)

        plt.close()

            
            
            
            





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
import torch 
import os 
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl 
import matplotlib.colors
import scenepic as sp 
from einops import rearrange

class Visualiser(pl.LightningModule):
    def __init__(self, save_dir, rank=0):
        super().__init__()
        self.save_dir = save_dir
        self.rank = rank

    def set_global_rank(self, global_rank):
        self.rank = global_rank

    def visualise_input_normal_imgs(self, normal_images):
        if self.rank == 0:
            B, N = normal_images.shape[:2]
            B = min(B, 2)
            N = min(N, 4)

            
            normal_images = np.transpose(normal_images, (0, 1, 3, 4, 2))

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
                    axes[b,n].imshow(normal_images[b,n])
                    # axes[b,n].axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_normal_images.png'))
            plt.close()
      
    def visualise_vc_as_image(
        self, 
        vc_pred, 
        vc=None, 
        mask=None, 
        conf=None, 
        plot_error_heatmap=True, 
        dvc=None, 
        dvc_pred=None, 
        vp_init_pred=None,
    ):
        B, N, H, W, C = vc_pred.shape

        B = min(B, 1)
        N = min(N, 4)

        if conf is not None:
            conf = 1 / conf 
            conf *= mask 

        if plot_error_heatmap:
            error_heatmap = np.linalg.norm(vc_pred - vc, axis=-1)
            error_heatmap *= mask
            error_heatmap[error_heatmap < 1e-4] = 1e-4
        
        if vc is not None:
            vc[~mask.astype(bool)] = 0

            norm_min, norm_max = vc.min(), vc.max()
            vc = (vc - norm_min) / (norm_max - norm_min) 
            vc[~mask.astype(bool)] = 1

        if mask is not None:
            vc_pred[~mask.astype(bool)] = 0
            vc_pred = (vc_pred - norm_min) / (norm_max - norm_min) 
            vc_pred[~mask.astype(bool)] = 1

            vc_pred = np.clip(vc_pred, 0, 1)


        if vp_init_pred is not None:
            vp_init_pred[~mask[:, None].repeat(N, axis=1).astype(bool)] = 0

            norm_min, norm_max = vp_init_pred.min(), vp_init_pred.max()
            vp_init_pred = (vp_init_pred - norm_min) / (norm_max - norm_min)
            vp_init_pred[~mask[:, None].repeat(N, axis=1).astype(bool)] = 1

        if dvc_pred is not None:
            dvc_pred = np.linalg.norm(dvc_pred, axis=-1)
        if dvc is not None:
            dvc = np.linalg.norm(dvc, axis=-1)


        num_rows = 2
        num_rows += 1 if plot_error_heatmap else 0
        num_rows += 1 if conf is not None else 0
        num_rows += 4 if vp_init_pred is not None else 0
        num_rows += 4 if dvc is not None else 0
        num_rows += 4 if dvc_pred is not None else 0
        fig = plt.figure(figsize=(4*N, num_rows*4*B))
        for b in range(B):
            for n in range(N):
                row = 0

                # ------------ gt vc ------------
                if vc is not None:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    plt.imshow(vc[b,n])
                    plt.title(f'Vc GT {n}')
                    plt.axis('off')
                    row += 1

                # ------------ pred vc ------------
                if vc_pred is not None:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    plt.imshow(vc_pred[b,n])
                    plt.title(f'Vc pred {n}')
                    plt.axis('off')
                    row += 1

                # ------------ error heatmap ------------
                if plot_error_heatmap:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    im = plt.imshow(error_heatmap[b,n], 
                                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('custom', [(1, 1, 1), (1, 0, 0)]), # white to red
                                    norm=matplotlib.colors.LogNorm(vmin=1e-4, vmax=max(1e-3, np.max(error_heatmap[b,n]))))
                    if n == N-1:
                        plt.colorbar(im)
                    plt.title(f'Error {n}')
                    plt.axis('off')
                    row += 1

                # ------------ conf ------------
                if conf is not None:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    im = plt.imshow(conf[b,n],
                                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list('custom', [(1, 1, 1), (1, 0.5, 0)]), # white to orange
                                    norm=matplotlib.colors.LogNorm(vmin=min(1e-3, np.min(conf[b,n]+1e-6)), vmax=max(1e-2, np.max(conf[b,n]))))
                    if n == N-1: 
                        plt.colorbar(im)
                    plt.title(f'1/conf {n}')
                    plt.axis('off')
                    row += 1

                # ------------ vp init pred ------------
                if vp_init_pred is not None:
                    for k in range(4):
                        plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                        im = plt.imshow(vp_init_pred[b,k,n])
                        plt.title(f'init $vp_{k}^{n}$')
                        plt.axis('off')
                        row += 1

                # ------------ dvc ------------
                if dvc is not None:
                    for k in range(4):
                        plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                        dvc_masked = dvc[b, k, n] * mask[b,n]
                        im = plt.imshow(dvc_masked, cmap='viridis')
                        plt.title(f'GT $dvc_{k}^{n}$')
                        if n == N-1:
                            plt.colorbar(im)
                        plt.axis('off')
                        row += 1

                # ------------ dvc pred ------------
                if dvc_pred is not None:
                    for k in range(4):
                        dvc_pred_masked = dvc_pred[b, k, n] * mask[b,n]
                        plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                        im = plt.imshow(dvc_pred_masked)
                        if n == N-1:
                            plt.colorbar(im)
                        plt.title(f'pred $dvc_{k}^{n}$')
                        plt.axis('off')
                        row += 1

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_pms.png'))
        plt.close()


    def visualise_vp_vc(
        self,
        vc, 
        vc_init_pred,
        vc_pred,
        vp,
        vp_init_pred,
        vp_pred,
        bg_mask,
        conf_mask,
        color,
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
        
        num_rows, num_cols = 3, 5
        sub_fig_size = 4 

        mask = (bg_mask * conf_mask).astype(bool)

        fig = plt.figure(figsize=(sub_fig_size * num_cols, sub_fig_size * num_rows))

        filtered_vc_pred = []
        filtered_colors = []

        ax = fig.add_subplot(num_rows, num_cols, 5, projection='3d')
        vc = rearrange(vc, 'b n h w c -> b n (h w) c')
        ax.scatter(vc[0, 0, :, 0], 
                   vc[0, 0, :, 1], 
                   vc[0, 0, :, 2], c='gray', s=s, alpha=gt_alpha, label=f'$vc_{0}$')
        
        for n in range(4):
            ax = fig.add_subplot(num_rows, num_cols, n+1, projection='3d')
            ax.scatter(vp[0, n, :, 0], 
                       vp[0, n, :, 1], 
                       vp[0, n, :, 2], c='gray', s=s, alpha=gt_alpha, label=f'$vp_{n}')

            filted_vp_init_pred = vp_init_pred[0, n].reshape(-1, 3)[mask[0].flatten()]
            ax = fig.add_subplot(num_rows, num_cols, 5+n+1, projection='3d')
            ax.scatter(filted_vp_init_pred[:, 0], 
                       filted_vp_init_pred[:, 1], 
                       filted_vp_init_pred[:, 2], c='red', s=s, alpha=pred_alpha, label=f'$vpinit_{n}$')
            
            filtered_vp_pred = vp_pred[0, n].reshape(-1, 3)[mask[0].flatten()]
            ax = fig.add_subplot(num_rows, num_cols, 10+n+1, projection='3d')
            ax.scatter(filtered_vp_pred[:, 0], 
                       filtered_vp_pred[:, 1], 
                       filtered_vp_pred[:, 2], c='orange', s=s, alpha=pred_alpha, label=f'$vp_{n}$')

        max_range = np.array([
            vc[0, :, :, 0].max() - vc[0, :, :, 0].min(),
            vc[0, :, :, 1].max() - vc[0, :, :, 1].min(),
            vc[0, :, :, 2].max() - vc[0, :, :, 2].min()
        ]).max() / 2.0
        mid_x = (vc[0, :, :, 0].max() + vc[0, :, :, 0].min()) * 0.5
        mid_y = (vc[0, :, :, 1].max() + vc[0, :, :, 1].min()) * 0.5
        mid_z = (vc[0, :, :, 2].max() + vc[0, :, :, 2].min()) * 0.5

        for ax in fig.axes:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.view_init(elev=10, azim=20, vertical_axis='y')
            ax.set_box_aspect([1, 1, 1])
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

        plt.tight_layout(pad=0.01)  # Reduce padding between subplots
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_vp_vc.png'), dpi=300)

        plt.close()
            
            






    def visualise(
        self, 
        normal_maps=None,
        vp=None, 
        vp_init_pred=None,
        vp_pred=None, 
        vc=None, 
        vc_init_pred=None,
        vc_pred=None, 
        dvc=None,
        dvc_pred=None,
        conf=None, 
        mask=None, 
        color=None, 
        vertex_visibility=None, 
        no_annotations=True,
        plot_error_heatmap=True,
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

        if conf is not None:
            conf_threshold = 0.08
            conf_mask = (1/conf) < conf_threshold
        
        self.visualise_input_normal_imgs(
            normal_maps
        )

        self.visualise_vc_as_image(
            vc_pred=vc_init_pred, #vc_pred, 
            vc=vc, 
            dvc=dvc,
            vp_init_pred=vp_init_pred,
            mask=mask, 
            conf=conf, 
            plot_error_heatmap=plot_error_heatmap,
            dvc_pred=dvc_pred
        )
        
        self.visualise_vp_vc(
            vp=vp, 
            vp_init_pred=vp_init_pred,
            vp_pred=vp_pred, 
            vc=vc, 
            vc_init_pred=vc_init_pred,
            vc_pred=vc_pred,
            bg_mask=mask, 
            conf_mask=conf_mask, 
            color=color, 
            # vertex_visibility=vertex_visibility, 
            # no_annotations=no_annotations
        )

        # import ipdb; ipdb.set_trace()
        
        # self.visualise_scenepic(
        #     vp=vp, 
        #     vc=vc, 
        #     vp_pred=rearrange(vp_pred, 'b n h w c -> b n (h w) c'), 
        #     vc_pred=rearrange(vc_pred, 'b n h w c -> b n (h w) c'), 
        #     vp_init_pred=rearrange(vp_init_pred, 'b n h w c -> b n (h w) c'),
        #     vc_init_pred=rearrange(vc_init_pred, 'b n h w c -> b n (h w) c'),
        #     color=color, 
        #     masks=mask, 
        #     vertex_visibility=vertex_visibility
        # )
        

    def visualise_scenepic(
        self,
        vp, 
        vc, 
        vp_pred, 
        vc_pred, 
        vp_init_pred=None,
        vc_init_pred=None,
        color=None, 
        masks=None, 
        vertex_visibility=None
    ):
        viridis = plt.colormaps.get_cmap('viridis')
        
        B, N = vp.shape[:2]
        B, N = min(B, 1), min(N, 4)

        scene = sp.Scene()


        # ----------------------- vc pred -----------------------
        positions = []
        colors = []
        for n in range(N):
            vc_plot = vc_pred[0, n, masks[0, n].astype(np.bool).flatten()].reshape(-1, 3)
            vc_plot[..., 0] += 2.0
            positions.append(vc_plot)
            colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())

        positions = np.concatenate(positions, axis=0)
        colors = np.concatenate(colors, axis=0)
        colors_normalized = colors / colors.max()
        colors_rgb = viridis(colors_normalized)[:, :3] 

        mesh_vc_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vc_pred")
        mesh_vc_pred.add_sphere() 
        mesh_vc_pred.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vc_pred.enable_instancing(positions = positions, colors = colors_rgb) 


        # ----------------------- vc gt -----------------------
        positions = []
        for n in range(N):
            vc_gt = vc[0, n, masks[0, n].astype(np.bool)].reshape(-1, 3)
            vc_gt[..., 0] += 2.0
            positions.append(vc_gt)
        positions = np.concatenate(positions, axis=0)

        mesh_vc_gt = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vc_gt")
        mesh_vc_gt.add_sphere() 
        mesh_vc_gt.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vc_gt.enable_instancing(positions = positions) 



        # ----------------------- vp pred -----------------------
        positions = []
        colors = []

        for n in range(N):
            vp_plot = vp_pred[0, n, masks[0, n].astype(np.bool).flatten()]
            vp_plot[..., 0] += 0.8 * n - 1.6
            positions.append(vp_plot)
            colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())


        positions = np.concatenate(positions, axis=0)
        colors = np.concatenate(colors, axis=0)


        colors_normalized = colors / colors.max()
        colors_rgb = viridis(colors_normalized)[:, :3] 

        mesh_vp_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vp_pred")
        mesh_vp_pred.add_sphere() 
        mesh_vp_pred.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vp_pred.enable_instancing(positions = positions, colors = colors_rgb) 


        # ----------------------- vp gt -----------------------
        positions = []
        for n in range(N):
            vp_gt_plot = vp[0, n, vertex_visibility[0, n].astype(np.bool).flatten()]
            vp_gt_plot[..., 0] += 0.8 * n - 1.6
            positions.append(vp_gt_plot)
        positions = np.concatenate(positions, axis=0)

        mesh_vp_gt = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vp_gt")
        mesh_vp_gt.add_sphere() 
        mesh_vp_gt.apply_transform(sp.Transforms.Scale(0.005)) 
        mesh_vp_gt.enable_instancing(positions = positions) 



        # ----------------------- vp init pred -----------------------
        if vp_init_pred is not None:

            # vp gt overlay 
            positions = []
            for n in range(N):
                vp_gt_plot = vp[0, n, vertex_visibility[0, n].astype(np.bool).flatten()]
                vp_gt_plot[..., 0] += 0.8 * n - 1.6
                vp_gt_plot[..., 1] -= 2.0
                positions.append(vp_gt_plot)
            positions = np.concatenate(positions, axis=0)

            mesh_vp_gt_row_2 = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vp_gt_row_2")
            mesh_vp_gt_row_2.add_sphere() 
            mesh_vp_gt_row_2.apply_transform(sp.Transforms.Scale(0.005)) 
            mesh_vp_gt_row_2.enable_instancing(positions = positions) 

            # vp init pred 
            positions = []
            colors = []
            for n in range(N):
                vp_init_pred_plot = vp_init_pred[0, n, masks[0, n].astype(np.bool).flatten()]
                vp_init_pred_plot[..., 0] += 0.8 * n - 1.6
                vp_init_pred_plot[..., 1] -= 2.0
                positions.append(vp_init_pred_plot)
                colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())
            positions = np.concatenate(positions, axis=0)
            colors = np.concatenate(colors, axis=0)
            colors_normalized = colors / colors.max()
            colors_rgb = viridis(colors_normalized)[:, :3] 

            mesh_vp_init_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vp_init_pred")
            mesh_vp_init_pred.add_sphere() 
            mesh_vp_init_pred.apply_transform(sp.Transforms.Scale(0.005)) 
            mesh_vp_init_pred.enable_instancing(positions = positions, colors = colors_rgb) 

        if vc_init_pred is not None:
            # vc gt overlay 
            positions = []
            for n in range(N):
                vc_gt = vc[0, n, masks[0, n].astype(np.bool)].reshape(-1, 3)
                vc_gt[..., 0] += 2.0
                positions.append(vc_gt)
            positions = np.concatenate(positions, axis=0)

            mesh_vc_gt_row_2 = scene.create_mesh(shared_color = sp.Colors.Gray, layer_id = "vc_gt_row_2")
            mesh_vc_gt_row_2.add_sphere() 
            mesh_vc_gt_row_2.apply_transform(sp.Transforms.Scale(0.005)) 
            mesh_vc_gt_row_2.enable_instancing(positions = positions) 



            positions = []
            colors = []
            for n in range(N):
                vc_init_pred_plot = vc_init_pred[0, n, masks[0, n].astype(np.bool).flatten()]
                vc_init_pred_plot[..., 0] += 2.0
                vc_init_pred_plot[..., 1] -= 2.0
                positions.append(vc_init_pred_plot)
                colors.append(color[0, n, masks[0, n].astype(np.bool)].flatten())
            positions = np.concatenate(positions, axis=0)
            colors = np.concatenate(colors, axis=0)
            colors_normalized = colors / colors.max()
            colors_rgb = viridis(colors_normalized)[:, :3] 

            mesh_vc_init_pred = scene.create_mesh(shared_color = sp.Color(0,1,0), layer_id = "vc_init_pred")
            mesh_vc_init_pred.add_sphere() 
            mesh_vc_init_pred.apply_transform(sp.Transforms.Scale(0.005)) 
            mesh_vc_init_pred.enable_instancing(positions = positions, colors = colors_rgb) 


        # ----------------------- canvas -----------------------
        golden_ratio = (1 + np.sqrt(5)) / 2
        canvas = scene.create_canvas_3d(width = 1600, height = 1600 / golden_ratio, shading=sp.Shading(bg_color=sp.Colors.White))
        frame = canvas.create_frame()

        frame.add_mesh(mesh_vp_pred)
        frame.add_mesh(mesh_vp_gt)

        frame.add_mesh(mesh_vc_pred)
        frame.add_mesh(mesh_vc_gt)

        if vp_init_pred is not None:
            frame.add_mesh(mesh_vp_init_pred)
            frame.add_mesh(mesh_vp_gt_row_2)
        if vc_init_pred is not None:
            frame.add_mesh(mesh_vc_init_pred)
            frame.add_mesh(mesh_vc_gt_row_2)

        path = os.path.join(self.save_dir, f'{self.global_step:06d}_sp.html')

        scene.save_as_html(path)
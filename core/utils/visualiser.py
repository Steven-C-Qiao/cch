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
        vp_init_pred=None
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
            vp_init_pred[~mask.astype(bool)] = 0

            norm_min, norm_max = vp_init_pred.min(), vp_init_pred.max()
            vp_init_pred = (vp_init_pred - norm_min) / (norm_max - norm_min)
            vp_init_pred[~mask.astype(bool)] = 1

        if dvc_pred is not None:
            dvc_pred = np.linalg.norm(dvc_pred, axis=-1)
        if dvc is not None:
            dvc = np.linalg.norm(dvc, axis=-1)


        num_rows = 2
        num_rows += 1 if plot_error_heatmap else 0
        num_rows += 1 if conf is not None else 0
        num_rows += 1 if dvc is not None else 0
        num_rows += 1 if vp_init_pred is not None else 0
        num_rows += 1 if dvc_pred is not None else 0
        fig = plt.figure(figsize=(4*N, num_rows*4*B))
        for b in range(B):
            for n in range(N):
                row = 0

                # ------------ gt vc ------------
                if vc is not None:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    plt.imshow(vc[b,n])
                    plt.title(f'GT Frame {n}')
                    plt.axis('off')
                    row += 1

                # ------------ pred vc ------------
                if vc_pred is not None:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    plt.imshow(vc_pred[b,n])
                    plt.title(f'Pred Frame {n}')
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
                    plt.title(f'Error Heatmap Frame {n}')
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
                    plt.title(f'1/conf Frame {n}')
                    plt.axis('off')
                    row += 1

                # ------------ vp init pred ------------
                if vp_init_pred is not None:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    im = plt.imshow(vp_init_pred[b,n])
                    plt.title(f'vp init pred no dvc {n}')
                    plt.axis('off')
                    row += 1

                # ------------ dvc ------------
                if dvc is not None:
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    dvc_masked = dvc[b,n] * mask[b,n]
                    im = plt.imshow(dvc_masked, cmap='viridis')
                    plt.title(f'dvc Frame {n}')
                    if n == N-1:
                        plt.colorbar(im)
                    plt.axis('off')
                    row += 1

                # ------------ dvc pred ------------
                if dvc_pred is not None:
                    dvc_pred_masked = dvc_pred[b,n] * mask[b,n]
                    plt.subplot(num_rows*B, N, (b*num_rows+row)*N + n + 1)
                    im = plt.imshow(dvc_pred_masked)
                    if n == N-1:
                        plt.colorbar(im)
                    plt.title(f'dvc_pred Frame {n}')
                    plt.axis('off')
                    row += 1


        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_colormaps.png'))
        plt.close()


    def visualise_vp_vc(
        self, 
        vp, 
        vc, 
        vp_pred, 
        vc_pred, 
        vp_init_pred=None, 
        vc_init_pred=None,
        bg_mask=None, 
        conf_mask=None,
        color=None, 
        vertex_visibility=None, 
        no_annotations=True
    ):
        """
        Visualise per-frame posed vertices and joined canonical points

        Args:
            vp: (B, N, 6890, 3) CAPE scan vertices
            vp_init_pred: (B, N, H * W, 3): w/o pose blendshapes
            vp_pred: (B, N, H * W, 3) predicted per-frame LBS posed vertices

            vc: (B, N, H, W, 3) per-frame canonical pointmap
            vc_init_pred: (B, N, H * W, 3): w/o pose blendshapes
            vc_pred: (B, N, H * W, 3) predicted per-frame canonical pointmap

            bg_mask: (B, N, H, W) background mask
            color: (B, N, H, W) argmax of skinning weights
            vertex_visibility: (B, N, 6890) vp visibility
            conf_mask: (B, N, H, W) 
            no_annotations: bool
        """
        s = 0.05
        gt_alpha = 0.3
        pred_alpha = 0.5
        B, N, V, C = vp.shape
        B, N = min(B, 1), min(N, 4)


        num_rows = B
        if vp_init_pred is not None:
            num_rows *= 2


        if conf_mask is not None and bg_mask is not None:
            mask = (bg_mask * conf_mask).astype(np.bool)
        elif conf_mask is not None:
            mask = conf_mask.astype(np.bool)
        elif bg_mask is not None:
            mask = bg_mask.astype(np.bool)
        else:
            mask = None

        fig = plt.figure(figsize=(4*(N+2), 4*num_rows))

        for b in range(B):
            all_vc_points = []
            all_vc_init_points = []
            all_colors = []
            

            for n in range(N):
                color_masked = color[b, n].flatten() if color is not None else 'red'
                    
                # ---------------- vc ----------------
                vc_pred_to_scatter = vc_pred[b, n]
                vp_pred_to_scatter = vp_pred[b, n]
                vp_to_scatter = vp[b, n]

                if vc_init_pred is not None:
                    vc_init_pred_to_scatter = vc_init_pred[b, n]

                if mask is not None:
                    vp_pred_to_scatter = vp_pred_to_scatter[mask[b, n].astype(np.bool).flatten()]
                    vc_pred_to_scatter = vc_pred_to_scatter[mask[b, n].astype(np.bool).flatten()]
                    vc_init_pred_to_scatter = vc_init_pred_to_scatter[mask[b, n].astype(np.bool).flatten()]
                    color_masked = color_masked[mask[b, n].astype(np.bool).flatten()]
                    
                all_vc_points.append(vc_pred_to_scatter)
                all_vc_init_points.append(vc_init_pred_to_scatter)
                all_colors.append(color_masked)


                # ---------------- vp final ----------------
                ax = fig.add_subplot(num_rows, N+2, (b)*(N+2) + n + 1, projection='3d')

                if vertex_visibility is not None:
                    vp_to_scatter = vp_to_scatter[vertex_visibility[b, n].astype(np.bool)]
                
                ax.scatter(vp_to_scatter[:,0], vp_to_scatter[:,1], vp_to_scatter[:,2], 
                           c='gray', s=0.75*s, alpha=gt_alpha, label='Ground Truth')
                
                ax.scatter(vp_pred_to_scatter[:,0], vp_pred_to_scatter[:,1], vp_pred_to_scatter[:,2],
                           c=color_masked, s=s, alpha=pred_alpha, label='Vp final pred')


                ax.view_init(elev=10, azim=20, vertical_axis='y')
                ax.set_box_aspect([1, 1, 1])
            
                max_range = np.array([
                    vp[b,n,:,0].max() - vp[b,n,:,0].min(),
                    vp[b,n,:,1].max() - vp[b,n,:,1].min(),
                    vp[b,n,:,2].max() - vp[b,n,:,2].min()
                ]).max() / 2.0
                mid_x = (vp[b,n,:,0].max() + vp[b,n,:,0].min()) * 0.5
                mid_y = (vp[b,n,:,1].max() + vp[b,n,:,1].min()) * 0.5
                mid_z = (vp[b,n,:,2].max() + vp[b,n,:,2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)



                # ---------------- vp init pred ----------------
                if vp_init_pred is not None:
                    ax = fig.add_subplot(num_rows, N+2, (1+b)*(N+2) + n + 1, projection='3d')

                    ax.scatter(vp_to_scatter[:,0], vp_to_scatter[:,1], vp_to_scatter[:,2], 
                            c='gray', s=0.75*s, alpha=gt_alpha, label='Ground Truth')
                
                    vp_init_pred_to_scatter = vp_init_pred[b, n]
                    vp_init_pred_to_scatter = vp_init_pred_to_scatter[mask[b, n].astype(np.bool).flatten()]
                    
                    ax.scatter(vp_init_pred_to_scatter[:,0], vp_init_pred_to_scatter[:,1], vp_init_pred_to_scatter[:,2], 
                               c=color_masked, s=s, alpha=pred_alpha, label='Vp init pred')
                        
                    ax.view_init(elev=10, azim=20, vertical_axis='y')
                    ax.set_box_aspect([1, 1, 1])
                
                    max_range = np.array([
                        vp[b,n,:,0].max() - vp[b,n,:,0].min(),
                        vp[b,n,:,1].max() - vp[b,n,:,1].min(),
                        vp[b,n,:,2].max() - vp[b,n,:,2].min()
                    ]).max() / 2.0
                    mid_x = (vp[b,n,:,0].max() + vp[b,n,:,0].min()) * 0.5
                    mid_y = (vp[b,n,:,1].max() + vp[b,n,:,1].min()) * 0.5
                    mid_z = (vp[b,n,:,2].max() + vp[b,n,:,2].min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)



            # ---------------- joined vc ----------------
            ax = fig.add_subplot(num_rows, N+2, b*(N+2) + N + 1, projection='3d')

            all_vc_points = np.concatenate(all_vc_points, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            ax.scatter(all_vc_points[:,0], all_vc_points[:,1], all_vc_points[:,2],
                    c=all_colors, s=s, alpha=pred_alpha, label='Canonical Points')
            
            ax.view_init(elev=10, azim=20, vertical_axis='y')
            ax.set_box_aspect([1, 1, 1])

            max_range = np.array([
                all_vc_points[:,0].max() - all_vc_points[:,0].min(),
                all_vc_points[:,1].max() - all_vc_points[:,1].min(),
                all_vc_points[:,2].max() - all_vc_points[:,2].min()
            ]).max() / 2.0
            mid_x = (all_vc_points[:,0].max() + all_vc_points[:,0].min()) * 0.5
            mid_y = (all_vc_points[:,1].max() + all_vc_points[:,1].min()) * 0.5
            mid_z = (all_vc_points[:,2].max() + all_vc_points[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # ---------------- joined vc gt ----------------
            ax = fig.add_subplot(num_rows, N+2, b*(N+2) + N + 2, projection='3d')

            vc_to_scatter = vc[b].reshape(-1, 3)
            ax.scatter(vc_to_scatter[:,0], vc_to_scatter[:,1], vc_to_scatter[:,2],
                    c='gray', s=s, alpha=gt_alpha, label='Canonical Points')
            
            ax.view_init(elev=10, azim=20, vertical_axis='y')
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            if vc_init_pred is not None:
                ax = fig.add_subplot(num_rows, N+2, (b+1)*(N+2) + N + 1, projection='3d')

                all_vc_init_points = np.concatenate(all_vc_init_points, axis=0)
                ax.scatter(all_vc_init_points[:,0], all_vc_init_points[:,1], all_vc_init_points[:,2],
                        c=all_colors, s=s, alpha=pred_alpha, label='Canonical Points init pred')
                
                ax.view_init(elev=10, azim=20, vertical_axis='y')
                ax.set_box_aspect([1, 1, 1])
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if no_annotations:
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
            normal_maps: (B, N, H, W, 3)

            vc: (B, N, H, W, 3)
            vc_init_pred: (B, N, H, W, 3): w/o pose blendshapes
            vc_pred: (B, N, H, W, 3)

            vp: (B, N, 6890, 3)
            vp_init_pred: (B, N, H, W, 3): w/o pose blendshapes
            vp_pred: (B, N, H, W, 3)
            
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
            vc_pred=vc_pred, 
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
            vp_init_pred=rearrange(vp_init_pred, 'b n h w c -> b n (h w) c'),
            vp_pred=rearrange(vp_pred, 'b n h w c -> b n (h w) c'), 
            vc=vc, 
            vc_init_pred=rearrange(vc_init_pred, 'b n h w c -> b n (h w) c'),
            vc_pred=rearrange(vc_pred, 'b n h w c -> b n (h w) c'), 
            bg_mask=mask, 
            color=color, 
            vertex_visibility=vertex_visibility, 
            conf_mask=conf_mask, 
            no_annotations=no_annotations
        )
        
        self.visualise_scenepic(
            vp=vp, 
            vc=vc, 
            vp_pred=rearrange(vp_pred, 'b n h w c -> b n (h w) c'), 
            vc_pred=rearrange(vc_pred, 'b n h w c -> b n (h w) c'), 
            vp_init_pred=rearrange(vp_init_pred, 'b n h w c -> b n (h w) c'),
            vc_init_pred=rearrange(vc_init_pred, 'b n h w c -> b n (h w) c'),
            color=color, 
            masks=mask, 
            vertex_visibility=vertex_visibility
        )
        

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
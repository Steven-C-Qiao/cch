import torch 
import os 
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl 
import matplotlib.colors
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
            # Convert to numpy and move channels to last dimension
            normal_images = normal_images.permute(0, 1, 3, 4, 2).cpu().detach().numpy()

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
                    axes[b,n].axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_normal_images.png'))
            plt.close()


    def visualise_vp(self, vp, vp_pred, mask=None, color=None, no_annotations=True):
        if self.rank == 0:
            B, N, V, C = vp.shape

            B = min(B, 2)
            N = min(N, 4)
            fig = plt.figure(figsize=(4*N, 4*B))

            for b in range(B):
                for n in range(N):
                    vp_pred_to_scatter = vp_pred[b, n]
                    if color is not None:
                        color_masked = color[b, n].flatten()
                    else:
                        color_masked = 'red'
                    if mask is not None:
                        vp_pred_to_scatter = vp_pred_to_scatter[mask[b, n, 0].astype(np.bool).flatten()]
                        color_masked = color_masked[mask[b, n, 0].astype(np.bool).flatten()]


                    # Create 3D subplot
                    ax = fig.add_subplot(B, N, b*N + n + 1, projection='3d')
                    
                    # Plot ground truth vertices in blue
                    ax.scatter(vp[b,n,:,0], vp[b,n,:,1], vp[b,n,:,2], 
                            c='gray', s=0.05, alpha=0.1, label='Ground Truth')
                    
                    # Plot predicted vertices in red
                    ax.scatter(vp_pred_to_scatter[:,0], vp_pred_to_scatter[:,1], vp_pred_to_scatter[:,2],
                            c=color_masked, s=0.5, alpha=0.5, label='Predicted')

                    # ax.set_title(f'Batch {b}, Frame {n}')
                    # ax.legend()

                    # set look into z direction
                    ax.view_init(elev=10, azim=20, vertical_axis='y')
                    
                    # Set equal aspect ratio
                    ax.set_box_aspect([1, 1, 1])
                    
                    # Set equal tick ratios
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

                    if no_annotations:
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

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{self.global_step}_vp.png'), dpi=300)
            plt.close()

    def visualise_vc(self, vc_pred, mask=None, color=None, no_annotations=False):
        """
        Visualise the canonical space
        """
        if self.rank == 0:
            B, N, V, C = vc_pred.shape

            B = min(B, 2)
            N = min(N, 4)
            
            fig = plt.figure(figsize=(4*B, 4*1))

            for b in range(B):
                ax = fig.add_subplot(1, B, b + 1, projection='3d')
                
                # Collect all points across N frames
                all_points = []
                all_colors = []
                for n in range(N):
                    vc_pred_to_scatter = vc_pred[b, n]
                    if color is not None:
                        color_masked = color[b, n].flatten()
                    else:
                        color_masked = 'blue'
                    if mask is not None:
                        vc_pred_to_scatter = vc_pred_to_scatter[mask[b, n, 0].astype(np.bool).flatten()]
                        color_masked = color_masked[mask[b, n, 0].astype(np.bool).flatten()]

                    all_points.append(vc_pred_to_scatter)
                    all_colors.append(color_masked)
                # Combine all points and plot them together
                all_points = np.concatenate(all_points, axis=0)
                all_colors = np.concatenate(all_colors, axis=0)
                ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2],
                        c=all_colors, s=0.5, alpha=0.5, label='Canonical Points')
                
                # set look into z direction
                ax.view_init(elev=10, azim=20, vertical_axis='y')

                # Remove all visual elements except points
                ax.set_box_aspect([1, 1, 1])
                if no_annotations:
                    ax.grid(False)
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.xaxis.pane.set_edgecolor('none')
                    ax.yaxis.pane.set_edgecolor('none') 
                    ax.zaxis.pane.set_edgecolor('none')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    # Remove the axis lines
                    ax.xaxis.line.set_color('none')
                    ax.yaxis.line.set_color('none')
                    ax.zaxis.line.set_color('none')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_vc.png'))
            plt.close()
                    
                    
    def visualise_vc_as_image(self, vc_pred, vc=None, mask=None, color=None, conf=None, plot_error_heatmap=True):
        if self.rank == 0:
            B, N, V, C = vc_pred.shape
            vc_pred = vc_pred.reshape(B, N, int(np.sqrt(V)), int(np.sqrt(V)), C)

            if conf is not None:
                conf = 1 / conf 
                conf *= mask 

            if plot_error_heatmap:
                error_heatmap = np.linalg.norm(vc_pred - vc, axis=-1)
                error_heatmap *= mask


            
            if vc is not None:
                vc[~mask.astype(bool)] = 0

                norm_min, norm_max = vc.min(), vc.max()
                vc = (vc - norm_min) / (norm_max - norm_min) 
                vc[~mask.astype(bool)] = 1

            if mask is not None:
                vc_pred[~mask.astype(bool)] = 0
                vc_pred = (vc_pred - norm_min) / (norm_max - norm_min) 
                vc_pred[~mask.astype(bool)] = 1

                # clip 
                vc_pred = np.clip(vc_pred, 0, 1)


            B = min(B, 2)
            N = min(N, 4)
            
            num_rows = 3 if plot_error_heatmap else 2
            num_rows += 1 if conf is not None else 0
            fig = plt.figure(figsize=(4*N, num_rows*4*B))

            for b in range(B):
                for n in range(N):
                    # Plot vc (ground truth) on top row
                    plt.subplot(num_rows*B, N, b*num_rows*N + n + 1)
                    if vc is not None:
                        plt.imshow(vc[b,n])
                        plt.title(f'GT Frame {n}')
                    plt.axis('off')

                    # Plot vc_pred (prediction) on bottom row 
                    plt.subplot(num_rows*B, N, (b*num_rows+1)*N + n + 1)
                    plt.imshow(vc_pred[b,n])
                    plt.title(f'Pred Frame {n}')
                    plt.axis('off')

                    if plot_error_heatmap:
                        # error_heatmap = np.linalg.norm(vc_pred[b,n] - vc[b,n], axis=-1)
                        # error_heatmap *= mask[b,n]
                        # error_heatmap[error_heatmap > 0.3] = 0
                        plt.subplot(num_rows*B, N, (b*num_rows+2)*N + n + 1)

                        colors = [(1, 1, 1), (1, 0, 0)]  # White to red
                        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors)
                        im = plt.imshow(error_heatmap[b,n], cmap=custom_cmap, 
                                        norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=np.max(error_heatmap[b,n])))
                        if n == N-1:
                            plt.colorbar(im)
                        plt.title(f'Error Heatmap Frame {n}')
                        plt.axis('off')
                    if conf is not None:
                        plt.subplot(num_rows*B, N, (b*num_rows+3)*N + n + 1)
                        # conf_masked = conf[b,n] * mask[b,n] + 1e-3
                        # conf_masked = conf_masked * mask[b,n]
                        colors = [(1, 1, 1), (1, 0.5, 0)]  # White to orange
                        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors)
                        im = plt.imshow(conf[b,n],
                                        cmap=custom_cmap,
                                        norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=np.max(conf[b,n])))
                        plt.title(f'1/conf Frame {n}')
                        if n == N-1: 
                            plt.colorbar(im)
                        plt.axis('off')    
                        
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_colormaps.png'))
            plt.close()


    def visualise_vp_vc(self, vp, vc, vp_pred, vc_pred, bg_mask=None, color=None, vertex_visibility=None, conf_mask=None,
                        no_annotations=True):
        """
        Visualise per-frame posed vertices and joined canonical points

        Args:
            vp: (B, N, 6890, 3) CAPE scan vertices
            vc: (B, N, H, W, 3) per-frame canonical pointmap
            vp_pred: (B, N, H * W, 3) predicted per-frame LBS posed vertices
            vc_pred: (B, N, H, W, 3) predicted per-frame canonical pointmap
            bg_mask: (B, N, H, W) background mask
            color: (B, N, H, W) argmax of skinning weights
            vertex_visibility: (B, N, 6890) vp visibility
            conf_mask: (B, N, H, W) 
            no_annotations: bool
        """
        if self.rank == 0:
            s = 0.04
            B, N, V, C = vp.shape
            B = min(B, 2)
            N = min(N, 4)

            if conf_mask is not None and bg_mask is not None:
                mask = (bg_mask * conf_mask ).astype(np.bool)
            elif conf_mask is not None:
                mask = conf_mask.astype(np.bool)
            elif bg_mask is not None:
                mask = bg_mask.astype(np.bool)
            else:
                mask = None

            fig = plt.figure(figsize=(4*(N+2), 4*B))

            for b in range(B):
                all_points = []
                all_colors = []
                vc_to_scatter = vc[b].reshape(-1, 3)

                for n in range(N):
                    color_masked = color[b, n].flatten() if color is not None else 'red'
                        
                    # ---------------- vc and skinning ----------------
                    vc_pred_to_scatter = vc_pred[b, n]
                    vp_pred_to_scatter = vp_pred[b, n]
                    vp_to_scatter = vp[b, n]

                    if mask is not None:
                        vp_pred_to_scatter = vp_pred_to_scatter[mask[b, n].astype(np.bool).flatten()]
                        vc_pred_to_scatter = vc_pred_to_scatter[mask[b, n].astype(np.bool).flatten()]
                        color_masked = color_masked[mask[b, n].astype(np.bool).flatten()]
                        
                    all_points.append(vc_pred_to_scatter)
                    all_colors.append(color_masked)

                    # ---------------- vp ----------------
                    ax = fig.add_subplot(B, N+2, b*(N+2) + n + 1, projection='3d')


                    if vertex_visibility is not None:
                        vp_to_scatter = vp_to_scatter[vertex_visibility[b, n].astype(np.bool)]

                    # ax.scatter(vp_to_scatter[:,0], vp_to_scatter[:,1], vp_to_scatter[:,2], 
                    #         c='red', s=0.2, alpha=0.5, label='Ground Truth')
                    
                    ax.scatter(vp_to_scatter[:,0], vp_to_scatter[:,1], vp_to_scatter[:,2], 
                            c='gray', s=0.75*s, alpha=0.5, label='Ground Truth')
                    
                    ax.scatter(vp_pred_to_scatter[:,0], vp_pred_to_scatter[:,1], vp_pred_to_scatter[:,2],
                            c=color_masked, s=s, alpha=0.5, label='Predicted')


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

                    # ax.set_title(f'Batch {b}, Frame {n}')
                    # ax.legend()

                # ---------------- joined vc ----------------
                ax = fig.add_subplot(B, N+2, b*(N+2) + N + 1, projection='3d')

                all_points = np.concatenate(all_points, axis=0)
                all_colors = np.concatenate(all_colors, axis=0)
                ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2],
                        c=all_colors, s=s, alpha=0.3, label='Canonical Points')
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

                # ---------------- joined vc gt ----------------
                ax = fig.add_subplot(B, N+2, b*(N+2) + N + 2, projection='3d')
                ax.scatter(vc_to_scatter[:,0], vc_to_scatter[:,1], vc_to_scatter[:,2],
                        c='gray', s=s, alpha=0.5, label='Canonical Points')
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
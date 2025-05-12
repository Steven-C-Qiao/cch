import torch 
import os 
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl 

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
            normal_images = normal_images.permute(0, 1, 3, 4, 2).cpu().numpy()

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
            plt.savefig(os.path.join(self.save_dir, 'normal_images.png'))
            plt.close()


    def visualise_vp(self, vp, vp_pred, mask=None, color=None):
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
                            c='blue', s=0.5, alpha=0.5, label='Ground Truth')
                    
                    # Plot predicted vertices in red
                    ax.scatter(vp_pred_to_scatter[:,0], vp_pred_to_scatter[:,1], vp_pred_to_scatter[:,2],
                            c=color_masked, s=0.5, alpha=0.3, label='Predicted')

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

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'vp_{self.global_step}.png'))
            plt.close()

    def visualise_vc(self, vc_pred, mask=None):
        """
        Visualise the canonical space
        """
        if self.rank == 0:
            B, N, V, C = vc_pred.shape

            B = min(B, 2)
            N = min(N, 4)
            
            fig = plt.figure(figsize=(4*1, 4*B))

            for b in range(B):
                ax = fig.add_subplot(B, 1, b + 1, projection='3d')
                
                # Collect all points across N frames
                all_points = []
                for n in range(N):
                    vc_pred_to_scatter = vc_pred[b, n]
                    if mask is not None:
                        vc_pred_to_scatter = vc_pred_to_scatter[mask[b, n, 0].astype(np.bool).flatten()]
                    all_points.append(vc_pred_to_scatter)
                
                # Combine all points and plot them together
                all_points = np.concatenate(all_points, axis=0)
                ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2],
                        c='blue', s=0.5, alpha=0.5, label='Canonical Points')
                
                # set look into z direction
                ax.view_init(elev=10, azim=20, vertical_axis='y')

                # Set equal aspect ratio
                ax.set_box_aspect([1, 1, 1])
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'vc_{self.global_step}.png'))
            plt.close()
                    
                    

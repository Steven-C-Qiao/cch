import torch 
import os 
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl 
import matplotlib.colors
import scenepic as sp 
from einops import rearrange

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Visualiser(pl.LightningModule):
    def __init__(self, save_dir, rank=0):
        super().__init__()
        self.save_dir = save_dir
        self.rank = rank

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

        self.visualise_vc_as_image(
            predictions,
            batch
        )

        self.visualise_vp_vc(
            predictions,
            batch
        )
        
        

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
      
    def visualise_vc_as_image(
        self, 
        predictions,
        batch
    ):
        B, N, H, W, C = predictions['vc_init'].shape
        mask = batch['masks']

        B = 1
        N = min(N, 4)
        sub_fig_size = 4
        num_cols = 4

        num_rows = 0
        if 'vc_init' in predictions:
            num_rows += 1
            vc_init = predictions['vc_init']
            vc_init[~mask.astype(bool)] = 0

            norm_min, norm_max = vc_init.min(), vc_init.max()
            vc_init = (vc_init - norm_min) / (norm_max - norm_min) 
            vc_init[~mask.astype(bool)] = 1

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

    
        fig = plt.figure(figsize=(num_cols*sub_fig_size, num_rows*sub_fig_size))
        for n in range(num_cols):
            row = 0

            if 'vc_maps' in batch:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(vc_maps[0, n])
                plt.title(f'Vc maps {n}')
                row += 1
                
            if 'vc_init' in predictions:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(vc_init[0, n])
                plt.title(f'Vc init {n}')
                row += 1

            if 'smpl_w_maps' in batch:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(smpl_w_maps[0, n])
                plt.title(f'Smpl w maps {n}')
                row += 1
            
            if "w" in predictions:
                plt.subplot(num_rows, num_cols, (row)*num_cols + n + 1)
                plt.imshow(w[0, n])
                plt.title(f'Pred w maps {n}')
                row += 1


        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_pms.png'))
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
        
        num_rows, num_cols = 3, 5
        sub_fig_size = 4 


        color = rearrange(batch['imgs'][0], 'n c h w -> (n h w) c')
        color = (color * IMAGENET_DEFAULT_STD) + IMAGENET_DEFAULT_MEAN
        color = color.astype(np.float32)


        fig = plt.figure(figsize=(sub_fig_size * num_cols, sub_fig_size * num_rows))

        ax = fig.add_subplot(num_rows, num_cols, 5, projection='3d')
        vc_gt = batch['template_mesh_verts'][0].cpu().detach().numpy()
        ax.scatter(vc_gt[:, 0], 
                   vc_gt[:, 1], 
                   vc_gt[:, 2], c='blue', s=s, alpha=gt_alpha, label=f'$vc_{0}$')

        if "vc_init" in predictions:
            ax = fig.add_subplot(num_rows, num_cols, num_cols+5, projection='3d')
            vc_init = predictions['vc_init']#.cpu().detach().numpy()
            vc_init = rearrange(vc_init, 'b n h w c -> b (n h w) c')
            ax.scatter(vc_init[0, :, 0], 
                       vc_init[0, :, 1], 
                       vc_init[0, :, 2], c=color, s=s, alpha=gt_alpha, label=f'$vc_init_{0}$')
            
        if "scan_mesh_verts_centered" in batch:
            scan_mesh_verts = batch['scan_mesh_verts_centered'][0]
            scan_mesh_colors = batch['scan_mesh_colors'][0]
            for i in range(4):
                verts = scan_mesh_verts[i].cpu().detach().numpy()
                colors = (scan_mesh_colors[i].cpu().detach().numpy() / 255.).astype(np.float32)
                ax = fig.add_subplot(num_rows, num_cols, i+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=colors, s=s, alpha=gt_alpha, label=f'$scan_{i}$')

        if "vp_init" in predictions:
            vp_init = predictions['vp_init'][0]#.cpu().detach().numpy()
            J_init = predictions['J_init'][0]#.cpu().detach().numpy()
            

            vp_init = rearrange(vp_init, 'k n h w c -> k (n h w) c')
            for i in range(4):
                verts = vp_init[i]
                ax = fig.add_subplot(num_rows, num_cols, num_cols+i+1, projection='3d')
                ax.scatter(verts[:, 0], 
                           verts[:, 1], 
                           verts[:, 2], c=color, s=s, alpha=gt_alpha, label=f'$vp_init_{i}$')
                ax.scatter(J_init[i, :, 0], 
                           J_init[i, :, 1], 
                           J_init[i, :, 2], c='green', s=1., alpha=gt_alpha, label=f'$J_init_{i}$')

        # ax = fig.add_subplot(num_rows, num_cols, 5+num_cols, projection='3d')
        # if vc_init_pred is not None:
        #     ax.scatter(vc_init_pred[0, 0, :, 0], 
        #             vc_init_pred[0, 0, :, 1], 
        #             vc_init_pred[0, 0, :, 2], c='red', s=s, alpha=gt_alpha, label=f'$vc_init_pred_{0}$')
            
        # for n in range(4):
        #     ax = fig.add_subplot(num_rows, num_cols, n+1, projection='3d')
        #     ax.scatter(vp[0, n, :, 0], 
        #                vp[0, n, :, 1], 
        #                vp[0, n, :, 2], c='gray', s=s, alpha=gt_alpha, label=f'$vp_{n}')

        #     filted_vp_init_pred = vp_init_pred[0, n].reshape(-1, 3)# [mask[0].flatten()]
        #     ax = fig.add_subplot(num_rows, num_cols, 5+n+1, projection='3d')
        #     ax.scatter(filted_vp_init_pred[:, 0], 
        #                filted_vp_init_pred[:, 1], 
        #                filted_vp_init_pred[:, 2], c='red', s=s, alpha=pred_alpha, label=f'$vpinit_{n}$')
            
        #     filtered_vp_pred = vp_pred[0, n].reshape(-1, 3)# [mask[0].flatten()]
        #     ax = fig.add_subplot(num_rows, num_cols, 10+n+1, projection='3d')
        #     ax.scatter(filtered_vp_pred[:, 0], 
        #                filtered_vp_pred[:, 1], 
        #                filtered_vp_pred[:, 2], c='orange', s=s, alpha=pred_alpha, label=f'$vp_{n}$')

        x = batch['template_mesh_verts'][0].cpu().detach().numpy()
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
        # self._no_annotations(fig)

        plt.tight_layout(pad=0.01)  # Reduce padding between subplots
        plt.savefig(os.path.join(self.save_dir, f'{self.global_step:06d}_vp_vc.png'), dpi=300)

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
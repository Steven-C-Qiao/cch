import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from scipy.spatial import cKDTree


try:
    import pyvista as pv
    PV_AVAILABLE = True
    pv.OFF_SCREEN = True
except ImportError:
    PV_AVAILABLE = False
    print("Warning: PyVista not available")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available")

def load_ply_vertices(ply_path):
    """Load vertices from PLY file."""
    if TRIMESH_AVAILABLE:
        mesh = trimesh.load(ply_path)
        if hasattr(mesh, 'vertices'):
            return mesh.vertices
    elif PV_AVAILABLE:
        mesh = pv.read(ply_path)
        return np.asarray(mesh.points)
    else:
        raise RuntimeError("Neither trimesh nor pyvista available to load PLY files")
    return None

def load_obj_vertices(obj_path):
    """Load vertices from OBJ file."""
    if TRIMESH_AVAILABLE:
        mesh = trimesh.load(obj_path)
        if hasattr(mesh, 'vertices'):
            return mesh.vertices
    elif PV_AVAILABLE:
        mesh = pv.read(obj_path)
        return np.asarray(mesh.points)
    else:
        raise RuntimeError("Neither trimesh nor pyvista available to load OBJ files")
    return None

def remove_outliers(points, k=20, threshold=5.0):
    """Remove statistical outliers using k-nearest neighbors statistics."""
    if len(points) < k + 1:
        return points
    
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    
    threshold_value = np.mean(mean_dist) + threshold * np.std(mean_dist)
    mask = mean_dist < threshold_value
    
    return points[mask]

def render_pointcloud_pyvista(verts, colors=None, center=None, camera_pos=None, max_range=None, point_size=0.5, alpha=1.0, default_color='gray', canvas_size=(1200, 800)):
    """Render point cloud to image using PyVista."""
    if not PV_AVAILABLE or len(verts) == 0:
        return np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    
    pcd = pv.PolyData(verts)
    plotter = pv.Plotter(off_screen=True, window_size=[canvas_size[1], canvas_size[0]])
    
    if colors is not None and len(colors) > 0:
        if colors.max() <= 1.0:
            colors_uint8 = (colors * 255).astype(np.uint8)
        else:
            colors_uint8 = colors.astype(np.uint8)
        pcd['colors'] = colors_uint8
        plotter.add_mesh(pcd, point_size=point_size, scalars='colors', rgb=True, opacity=alpha)
    else:
        print(f'Warning: No colors provided for point cloud')
        plotter.add_mesh(pcd, point_size=point_size, color=default_color, opacity=alpha)
    
    if center is not None and camera_pos is not None:
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]
        plotter.camera.view_angle = 30.0
    elif center is not None:
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]
        plotter.camera.view_angle = 30.0
    
    img = plotter.screenshot(None, return_img=True, transparent_background=True)
    plotter.close()
    return img

def get_view_params(x):
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
    
    elev, azim = 10, 20
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    distance = max_range * 4.0
    
    camera_offset = np.array([
        distance * np.cos(elev_rad) * np.sin(azim_rad),
        distance * np.sin(elev_rad),
        distance * np.cos(elev_rad) * np.cos(azim_rad)
    ])
    camera_pos = center + camera_offset
    
    return center, camera_pos, max_range

def compute_all_distances(up2you_path, ours_path, stride=5, device='cuda', rows_to_plot=['GT', 'UP2You', 'Ours']):
    """Compute chamfer distances for all pairs and return global min/max."""
    all_distances = []
    up2you_data = []
    ours_data = []
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Process UP2You data
    if 'UP2You' in rows_to_plot and os.path.exists(up2you_path):
        output_dirs = [d for d in os.listdir(up2you_path) if d.startswith('outputs_')]
        def extract_idx(dirname):
            try:
                return int(dirname.replace('outputs_', ''))
            except:
                return -1
        output_dirs = sorted(output_dirs, key=extract_idx)[::stride]
        
        for output_dir in output_dirs:
            gt_mesh_path = os.path.join(up2you_path, output_dir, 'meshes', 'gt_mesh_aligned.obj')
            pred_mesh_path = os.path.join(up2you_path, output_dir, 'meshes', 'pred_mesh_aligned.obj')
            
            if not os.path.exists(gt_mesh_path) or not os.path.exists(pred_mesh_path):
                up2you_data.append(None)
                continue
            
            try:
                gt_vertices = load_obj_vertices(gt_mesh_path)
                pred_vertices = load_obj_vertices(pred_mesh_path)
                
                gt_vertices_t = torch.tensor(gt_vertices, dtype=torch.float32).unsqueeze(0).to(device)
                pred_vertices_t = torch.tensor(pred_vertices, dtype=torch.float32).unsqueeze(0).to(device)
                
                gt_ptcld = Pointclouds(points=gt_vertices_t)
                pred_ptcld = Pointclouds(points=pred_vertices_t)
                
                errors, _ = chamfer_distance(pred_ptcld, gt_ptcld, batch_reduction=None, point_reduction=None)
                error_pred2gt = errors[0]
                distance_pred2gt = torch.sqrt(error_pred2gt) * 100.0
                
                # Squeeze to remove batch dimension: (1, N) -> (N,)
                distance_np = distance_pred2gt.squeeze(0).cpu().numpy()
                all_distances.append(distance_np)
                up2you_data.append({
                    'gt_vertices': gt_vertices,
                    'pred_vertices': pred_vertices,
                    'distance': distance_np,
                    'idx': output_dir.replace('outputs_', '')
                })
            except Exception as e:
                print(f'Error processing UP2You {output_dir}: {e}')
                up2you_data.append(None)
    
    # Process Ours data
    if 'Ours' in rows_to_plot and os.path.exists(ours_path):
        all_files = os.listdir(ours_path)
        gt_files = sorted([f for f in all_files if f.startswith('gt_vp_') and f.endswith('.ply')])
        
        for gt_file in gt_files[::stride]:
            timestep = gt_file.replace('gt_vp_', '').replace('.ply', '')
            pred_file = f'pred_vp_{timestep}.ply'
            
            gt_ply_path = os.path.join(ours_path, gt_file)
            pred_ply_path = os.path.join(ours_path, pred_file)
            
            if not os.path.exists(gt_ply_path) or not os.path.exists(pred_ply_path):
                ours_data.append(None)
                continue
            
            try:
                gt_vertices = load_ply_vertices(gt_ply_path)
                pred_vertices = load_ply_vertices(pred_ply_path)
                
                # Remove outliers from pred
                pred_vertices_original = pred_vertices.copy()
                pred_vertices = remove_outliers(pred_vertices, k=20, threshold=5.0)
                if len(pred_vertices) < 100:
                    pred_vertices = pred_vertices_original
                
                gt_vertices_t = torch.tensor(gt_vertices, dtype=torch.float32).unsqueeze(0).to(device)
                pred_vertices_t = torch.tensor(pred_vertices, dtype=torch.float32).unsqueeze(0).to(device)
                
                gt_ptcld = Pointclouds(points=gt_vertices_t)
                pred_ptcld = Pointclouds(points=pred_vertices_t)
                
                errors, _ = chamfer_distance(pred_ptcld, gt_ptcld, batch_reduction=None, point_reduction=None)
                error_pred2gt = errors[0]
                distance_pred2gt = torch.sqrt(error_pred2gt) * 100.0
                
                # Squeeze to remove batch dimension: (1, N) -> (N,)
                distance_np = distance_pred2gt.squeeze(0).cpu().numpy()
                all_distances.append(distance_np)
                ours_data.append({
                    'gt_vertices': gt_vertices,
                    'pred_vertices': pred_vertices,
                    'distance': distance_np,
                    'timestep': timestep
                })
            except Exception as e:
                print(f'Error processing Ours {timestep}: {e}')
                ours_data.append(None)
    
    # Compute global min/max
    if len(all_distances) > 0:
        all_distances_flat = np.concatenate([d.flatten() for d in all_distances])
        global_min = all_distances_flat.min()
        global_max = all_distances_flat.max()
    else:
        global_min, global_max = 0.0, 1.0
    
    print(f'Global distance range: [{global_min:.4f}, {global_max:.4f}] cm')
    
    # Collect GT data for rendering
    # Use GT from Ours path (gt_vp_*.ply files) as they match the timesteps
    gt_data = []
    
    if 'GT' in rows_to_plot and os.path.exists(ours_path):
        all_files = os.listdir(ours_path)
        gt_files = [f for f in all_files if f.startswith('gt_vp_') and f.endswith('.ply')]
        
        # Sort numerically by timestep
        def extract_timestep(filename):
            try:
                return int(filename.replace('gt_vp_', '').replace('.ply', ''))
            except:
                return -1
        gt_files = sorted(gt_files, key=extract_timestep)
        
        for gt_file in gt_files[::stride]:
            timestep = gt_file.replace('gt_vp_', '').replace('.ply', '')
            gt_ply_path = os.path.join(ours_path, gt_file)
            
            if not os.path.exists(gt_ply_path):
                gt_data.append(None)
                continue
            
            try:
                gt_vertices = load_ply_vertices(gt_ply_path)
                gt_data.append({
                    'gt_vertices': gt_vertices,
                    'timestep': timestep
                })
            except Exception as e:
                print(f'Error loading GT {timestep}: {e}')
                gt_data.append(None)
    
    return up2you_data, ours_data, gt_data, global_min, global_max

def render_gt_pointcloud(gt_vertices, point_size=0.7, canvas_size=(1200, 800)):
    """Render GT point cloud without error coloring."""
    if gt_vertices is None:
        return None
    
    # Get view parameters
    center, camera_pos, max_range = get_view_params(gt_vertices)
    
    # Render without colors (default gray)
    rendered_img = render_pointcloud_pyvista(
        verts=gt_vertices,
        colors=None,
        center=center,
        camera_pos=camera_pos,
        max_range=max_range,
        point_size=point_size,
        alpha=1.0,
        default_color='gray',
        canvas_size=canvas_size
    )
    
    # Keep RGBA format for transparent background
    if rendered_img.shape[2] == 3:
        rendered_img = np.concatenate([rendered_img, np.full((*rendered_img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
    
    return rendered_img

def render_with_global_scale(data, global_min, global_max, colormap, point_size=0.7, canvas_size=(1200, 800)):
    """Render point cloud with global color scale."""
    if data is None:
        return None, None
    
    distance = data['distance']
    pred_vertices = data['pred_vertices']
    
    # Flatten distance array to match number of points
    if distance.ndim > 1:
        distance = distance.flatten()
    
    # Ensure distance matches number of vertices
    if len(distance) != len(pred_vertices):
        print(f'Warning: Distance length ({len(distance)}) does not match vertices ({len(pred_vertices)})')
        # If distance is shorter, pad with last value; if longer, truncate
        if len(distance) < len(pred_vertices):
            distance = np.pad(distance, (0, len(pred_vertices) - len(distance)), mode='edge')
        else:
            distance = distance[:len(pred_vertices)]
    
    # Normalize using global scale
    distance_normalized = (distance - global_min) / (global_max - global_min + 1e-8)
    distance_normalized = np.clip(distance_normalized, 0, 1)
    
    # Apply colormap - colormap expects 1D array and returns (N, 4) for RGBA
    if distance_normalized.ndim == 0:
        # Single value case
        colors = colormap(distance_normalized)[:3].reshape(1, 3)
    else:
        colors = colormap(distance_normalized)[:, :3]
    
    # Get view parameters
    center, camera_pos, max_range = get_view_params(pred_vertices)
    
    # Render
    rendered_img = render_pointcloud_pyvista(
        verts=pred_vertices,
        colors=colors,
        center=center,
        camera_pos=camera_pos,
        max_range=max_range,
        point_size=point_size,
        alpha=1.0,
        default_color='gray',
        canvas_size=canvas_size
    )
    
    # Keep RGBA format for transparent background
    # PyVista already returns RGBA with transparent background
    if rendered_img.shape[2] == 3:
        # If RGB, convert to RGBA with full opacity
        rendered_img = np.concatenate([rendered_img, np.full((*rendered_img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
    
    return rendered_img, distance

def create_comparison_figure(gt_images, up2you_images, ours_images, gt_labels, up2you_labels, ours_labels, global_min, global_max, rows_to_plot=['GT', 'UP2You', 'Ours'], output_path='Figures/vis/00191/00191_Take2_exp_100_2_error_comparison.png', save_dpi=300):
    """Create a figure with selected rows comparing results with 30% overlap."""
    num_subplots = len(gt_images) if gt_images else (len(up2you_images) if up2you_images else len(ours_images))
    if num_subplots == 0:
        print("No images to plot!")
        return
    
    overlap_fraction = 0.3
    subfig_size = 4
    num_rows = len(rows_to_plot)
    
    # Create figure with selected number of rows
    fig = plt.figure(figsize=(subfig_size * num_subplots, subfig_size * num_rows * 1.4))
    fig.patch.set_alpha(0)  # Make figure background transparent
    
    # Map row names to their data
    row_data = {
        'GT': {'images': gt_images, 'labels': gt_labels, 'label': 'GT'},
        'UP2You': {'images': up2you_images, 'labels': up2you_labels, 'label': 'UP2You'},
        'Ours': {'images': ours_images, 'labels': ours_labels, 'label': 'Ours'}
    }
    
    # Create axes for each selected row
    row_axes_dict = {}
    for row_idx, row_name in enumerate(rows_to_plot):
        if row_name not in row_data:
            continue
        
        data = row_data[row_name]
        axes_list = []
        
        for idx in range(num_subplots):
            ax = fig.add_subplot(num_rows, num_subplots, row_idx * num_subplots + idx + 1)
            
            if idx < len(data['images']) and data['images'][idx] is not None:
                ax.imshow(data['images'][idx])  # matplotlib handles RGBA automatically
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            
            ax.axis('off')
            ax.patch.set_alpha(0)  # Make axis background transparent
            axes_list.append(ax)
        
        if len(axes_list) > 0:
            row_axes_dict[row_name] = axes_list
    
    # Apply tight layout first, then adjust rows with overlap
    plt.tight_layout(pad=0.1)
    
    # Apply overlap to all rows
    try:
        for row_name in rows_to_plot:
            axes_list = row_axes_dict.get(row_name, [])
            if len(axes_list) > 1:
                # Get position of first axis
                pos0 = axes_list[0].get_position()
                width = pos0.width
                height = pos0.height
                y0 = pos0.y0
                x0 = pos0.x0
                step = width * (1.0 - overlap_fraction)
                for i, ax in enumerate(axes_list):
                    new_x0 = x0 + i * step
                    ax.set_position([new_x0, y0, width, height])
    except Exception as e:
        print(f"Error in applying overlap: {e}")
        pass
    
    # Add row labels at the top-left corner of each row
    try:
        label_fontsize = 20
        label_offset_x = 0.005
        label_offset_y = 0.01
        for row_name in rows_to_plot:
            axes_list = row_axes_dict.get(row_name, [])
            if len(axes_list) > 0:
                data = row_data.get(row_name, {})
                label_text = data.get('label', row_name)
                pos = axes_list[0].get_position()
                x = pos.x0 + label_offset_x
                y = pos.y1 - label_offset_y
                fig.text(x, y, label_text, va='top', ha='left', fontsize=label_fontsize, fontweight='bold')
    except Exception as e:
        print(f"Error adding row labels: {e}")
        pass
    
    # Add global colorbar at the right edge (thinner)
    try:
        colormap = plt.colormaps.get_cmap('RdYlGn_r')
    except AttributeError:
        colormap = plt.cm.get_cmap('RdYlGn_r')
    
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    
    # Position colorbar at the right edge (spans all rows)
    # Find the first and last rows to determine colorbar height
    first_row_axes = None
    last_row_axes = None
    for row_name in rows_to_plot:
        axes_list = row_axes_dict.get(row_name, [])
        if len(axes_list) > 0:
            if first_row_axes is None:
                first_row_axes = axes_list
            last_row_axes = axes_list
    
    if last_row_axes and len(last_row_axes) > 0:
        pos_last = last_row_axes[-1].get_position()
        right_edge = pos_last.x0 + pos_last.width
        cbar_width = 0.008  # Thinner colorbar
        
        if first_row_axes and len(first_row_axes) > 0:
            cbar_height = (first_row_axes[0].get_position().y0 + first_row_axes[0].get_position().height) - last_row_axes[0].get_position().y0
            cbar_y0 = last_row_axes[0].get_position().y0
        else:
            cbar_height = last_row_axes[0].get_position().height
            cbar_y0 = last_row_axes[0].get_position().y0
        
        cbar_ax = fig.add_axes([right_edge + 0.01, cbar_y0, cbar_width, cbar_height])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Distance Error (cm)', rotation=270, labelpad=20, fontsize=12)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=save_dpi, bbox_inches='tight', pad_inches=0.1, transparent=True)
    print(f'Saved comparison figure to: {output_path}')
    plt.close()


def main(up2you_path, ours_path, stride, rows_to_plot=['GT', 'UP2You', 'Ours'], output_path='Figures/error_comparison.png', save_dpi=300, canvas_size=(1200, 800), point_size=2.0):
    """Main function to create error comparison figure.
    
    Args:
        up2you_path: Path to UP2You results directory
        ours_path: Path to Ours results directory
        stride: Stride for selecting samples (e.g., 5 means every 5th sample)
        rows_to_plot: List of rows to plot, e.g., ['GT', 'UP2You', 'Ours'] or ['GT', 'Ours']
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Plotting rows: {rows_to_plot}')
    
    print('Computing all distances to find global scale...')
    up2you_data, ours_data, gt_data, global_min, global_max = compute_all_distances(
        up2you_path, ours_path, stride=stride, device=device, rows_to_plot=rows_to_plot
    )
    
    print(f'Rendering with global scale [{global_min:.4f}, {global_max:.4f}] cm...')
    # Use new matplotlib API to avoid deprecation warning
    try:
        colormap = plt.colormaps.get_cmap('RdYlGn_r')
    except AttributeError:
        colormap = plt.cm.get_cmap('RdYlGn_r')
    
    # Render GT images (no error coloring)
    gt_images = []
    gt_labels = []
    if 'GT' in rows_to_plot:
        for data in gt_data:
            if data is not None:
                img = render_gt_pointcloud(data['gt_vertices'], point_size=point_size, canvas_size=canvas_size)
                gt_images.append(img)
                # Use timestep if available, otherwise use idx
                label = data.get('timestep', data.get('idx', ''))
                gt_labels.append(f'GT {label}')
            else:
                gt_images.append(None)
                gt_labels.append('')
    
    up2you_images = []
    ours_images = []
    up2you_labels = []
    ours_labels = []
    
    # Render UP2You images
    if 'UP2You' in rows_to_plot:
        for data in up2you_data:
            if data is not None:
                img, _ = render_with_global_scale(data, global_min, global_max, colormap, point_size=point_size, canvas_size=canvas_size)
                up2you_images.append(img)
                up2you_labels.append(f'UP2You {data["idx"]}')
            else:
                up2you_images.append(None)
                up2you_labels.append('')
    
    # Render Ours images
    if 'Ours' in rows_to_plot:
        for data in ours_data:
            if data is not None:
                img, _ = render_with_global_scale(data, global_min, global_max, colormap, point_size=point_size, canvas_size=canvas_size)
                ours_images.append(img)
                ours_labels.append(f'Ours {data["timestep"]}')
            else:
                ours_images.append(None)
                ours_labels.append('')
    
    # No padding needed - data is already strided from compute_all_distances
    
    print(f'Rendered {sum(1 for img in gt_images if img is not None)} GT images')
    print(f'Rendered {sum(1 for img in up2you_images if img is not None)} UP2You images')
    print(f'Rendered {sum(1 for img in ours_images if img is not None)} Ours images')
    
    # Compute and print mean chamfer distances
    up2you_means = []
    for data in up2you_data:
        if data is not None:
            mean_dist = data['distance'].mean()
            up2you_means.append(mean_dist)
    
    ours_means = []
    for data in ours_data:
        if data is not None:
            mean_dist = data['distance'].mean()
            ours_means.append(mean_dist)
    
    if len(up2you_means) > 0:
        up2you_overall_mean = np.mean(up2you_means)
        print(f'\nUP2You mean chamfer distance: {up2you_overall_mean:.4f} cm (across {len(up2you_means)} samples)')
    else:
        print('\nUP2You: No valid data')
    
    if len(ours_means) > 0:
        ours_overall_mean = np.mean(ours_means)
        print(f'Ours mean chamfer distance: {ours_overall_mean:.4f} cm (across {len(ours_means)} samples)')
    else:
        print('Ours: No valid data')
    
    print('Creating comparison figure...')
    create_comparison_figure(gt_images, up2you_images, ours_images, gt_labels, up2you_labels, ours_labels, global_min, global_max, rows_to_plot=rows_to_plot, output_path=output_path, save_dpi=save_dpi)
    print('Done!')

    ''' ---------------------------------------------------------- '''
if __name__ == '__main__':
    save_dpi = 200
    canvas_size = (900, 600)
    point_size = 3.0

    # Configuration: Select which rows to plot
    # Options: 'GT', 'UP2You', 'Ours'
    rows_to_plot = ['GT', 'UP2You', 'Ours']  # Change this to select specific rows, e.g., ['GT', 'Ours'] or ['UP2You']
    
    # Paths to the visualization directories
    # up2you_path = 'Figures/vis/00191/00191_Take2_exp_100_2/'
    # ours_path = 'Figures/vis/00191/00191_Take2_exp_100_2/'

    # output_path = 'Figures/vis/00191/00191_Take2_exp_100_2_error_comparison.png'

    up2you_path = 'Figures/up2you_vis/00134_Take1/'
    ours_path = 'Figures/vis/00134/00134_Take1_exp_104_ep9/'

    save_path = 'Figures/vis/00134/00134_Take1_exp_104_ep9_error.png'

    # Number of subplots per row
    stride = 5

    main(up2you_path, ours_path, stride=stride, rows_to_plot=rows_to_plot, output_path=save_path, save_dpi=save_dpi, canvas_size=canvas_size, point_size=point_size)
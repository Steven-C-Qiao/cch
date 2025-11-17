import os
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds

output_path = 'Figures/up2you_vis/00134_take1/'

try:
    import pyvista as pv
    PV_AVAILABLE = True
    # Set PyVista to use offscreen rendering
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

def render_pointcloud_pyvista(verts, colors=None, center=None, camera_pos=None, max_range=None, point_size=1., alpha=0.5, default_color='gray'):
    """Render point cloud to image using PyVista with y-axis up, elev=10, azim=20"""
    if not PV_AVAILABLE or len(verts) == 0:
        # Return blank image
        img = np.zeros((600, 400, 3), dtype=np.uint8)
        return img
    
    # Create point cloud
    pcd = pv.PolyData(verts)
    
    # Create plotter with explicit offscreen mode (portrait window, larger resolution)
    plotter = pv.Plotter(off_screen=True, window_size=[400, 600])
    
    # PyVista expects colors as uint8
    if colors is not None and len(colors) > 0:
        if colors.max() <= 1.0:
            colors_uint8 = (colors * 255).astype(np.uint8)
        else:
            colors_uint8 = colors.astype(np.uint8)
        pcd['colors'] = colors_uint8
        plotter.add_mesh(pcd, point_size=point_size, scalars='colors', 
                        rgb=True, opacity=alpha)
    else:
        # Use default color for point clouds without colors (e.g., GT scans)
        plotter.add_mesh(pcd, point_size=point_size, color=default_color, opacity=alpha)
    
    # Set camera with y-axis up, elev=10, azim=20
    if center is not None and camera_pos is not None:
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]  # y-axis up
        plotter.camera.view_angle = 30.0  # Fixed field of view angle
    elif center is not None:
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]  # y-axis up
        plotter.camera.view_angle = 30.0  # Fixed field of view angle
    
    # Render to numpy array
    # Capture with transparent background (returns RGBA)
    img = plotter.screenshot(None, return_img=True, transparent_background=True)
    plotter.close()
    
    return img

def render_mesh_pyvista(verts, faces, colors=None, center=None, camera_pos=None, max_range=None, opacity=1.0, default_color='lightgray'):
    """Render triangle mesh to image using PyVista with y-axis up, elev=10, azim=20."""
    if not PV_AVAILABLE or len(verts) == 0 or faces is None or len(faces) == 0:
        # Fallback to point rendering if mesh not available
        return render_pointcloud_pyvista(verts, colors=colors, center=center, camera_pos=camera_pos, max_range=max_range, point_size=1.5, alpha=opacity, default_color=default_color)
    # Build PolyData from verts and triangular faces
    faces_tri = np.asarray(faces, dtype=np.int64)
    if faces_tri.ndim != 2 or faces_tri.shape[1] != 3:
        # Attempt to coerce to triangles
        try:
            faces_tri = faces_tri[:, :3]
        except Exception:
            return render_pointcloud_pyvista(verts, colors=colors, center=center, camera_pos=camera_pos, max_range=max_range, point_size=1.5, alpha=opacity, default_color=default_color)
    # PyVista face array format: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    num_faces = faces_tri.shape[0]
    faces_pv = np.hstack([np.full((num_faces, 1), 3, dtype=np.int64), faces_tri]).reshape(-1)
    mesh = pv.PolyData(verts, faces_pv)
    plotter = pv.Plotter(off_screen=True, window_size=[400, 600])
    if colors is not None and len(colors) > 0:
        if colors.max() <= 1.0:
            colors_uint8 = (colors * 255).astype(np.uint8)
        else:
            colors_uint8 = colors.astype(np.uint8)
        mesh['colors'] = colors_uint8
        plotter.add_mesh(mesh, scalars='colors', rgb=True, opacity=opacity)
    else:
        plotter.add_mesh(mesh, color=default_color, opacity=opacity)
    if center is not None and camera_pos is not None:
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]
        plotter.camera.view_angle = 30.0
    elif center is not None:
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]
        plotter.camera.view_angle = 30.0
    # Capture with transparent background (returns RGBA)
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
    
    # y-axis up coordinate system: x=right, y=up, z=forward/back
    # elev=10: elevation angle from horizontal (x-z plane)
    # azim=20: rotation around y-axis
    elev, azim = 10, 20
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    distance = max_range * 5.0  # Increased distance to move camera farther away
    
    # Calculate camera position (from center, looking towards center)
    camera_offset = np.array([
        distance * np.cos(elev_rad) * np.sin(azim_rad),
        distance * np.sin(elev_rad),
        distance * np.cos(elev_rad) * np.cos(azim_rad)
    ])
    camera_pos = center + camera_offset
    
    # Front vector points from camera to center (opposite of camera offset)
    front_vec = -camera_offset / np.linalg.norm(camera_offset)
    
    return center, camera_pos, max_range, front_vec

if __name__ == '__main__':
    # Detect device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    for output_dir in sorted(os.listdir(output_path)):
        gt_mesh_path = os.path.join(output_path, output_dir, 'meshes', 'gt_mesh_aligned.obj')
        pred_mesh_path = os.path.join(output_path, output_dir, 'meshes', 'pred_mesh_aligned.obj')
        gt_mesh = trimesh.load(gt_mesh_path)
        pred_mesh = trimesh.load(pred_mesh_path)
        
        # Convert vertices to torch tensors and move to device
        gt_vertices = torch.tensor(gt_mesh.vertices, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)
        pred_vertices = torch.tensor(pred_mesh.vertices, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)
        
        gt_ptcld = Pointclouds(points=gt_vertices)
        pred_ptcld = Pointclouds(points=pred_vertices)
        errors, _ = chamfer_distance(
            pred_ptcld, 
            gt_ptcld, 
            batch_reduction=None, 
            point_reduction=None
        )
        error_pred2gt = errors[0]
        error_gt2pred = errors[1]
        
        distance_pred2gt = torch.sqrt(error_pred2gt) * 100.0 # Shape: (1, N)
        distance_gt2pred = torch.sqrt(error_gt2pred) * 100.0

        # Normalize distance values to [0, 1] for colormap
        distance_np = distance_pred2gt.squeeze(0).cpu().numpy()  # Remove batch dim: (N,)
        if distance_np.max() > distance_np.min():
            distance_normalized = (distance_np - distance_np.min()) / (distance_np.max() - distance_np.min())
        else:
            distance_normalized = np.zeros_like(distance_np)
        
        # Create green-to-red colormap with yellow transition (green=0, red=1)
        # Using matplotlib's RdYlGn_r (Red-Yellow-Green reversed) for green->yellow->red
        colormap = plt.cm.get_cmap('RdYlGn_r')  # Reversed: green at low values, yellow in middle, red at high
        colors = colormap(distance_normalized)[:, :3]  # Get RGB, ignore alpha: (N, 3)
        colors_tensor = torch.tensor(colors, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 3)
        
        # Assign colors to pred_ptcld using features (on device)
        print(pred_vertices.shape, colors_tensor.shape)
        pred_ptcld = Pointclouds(
            points=pred_vertices,
            features=colors_tensor
        )
        
        # Extract vertices and colors for rendering (move to CPU)
        pred_verts_np = pred_vertices.squeeze(0).cpu().numpy()  # (N, 3)
        pred_colors_np = colors  # Already numpy array (N, 3)
        
        # Get view parameters for consistent camera setup
        center, camera_pos, max_range, front_vec = get_view_params(pred_verts_np)
        
        # Render the colored point cloud
        rendered_img = render_pointcloud_pyvista(
            verts=pred_verts_np,
            colors=pred_colors_np,
            center=center,
            camera_pos=camera_pos,
            max_range=max_range,
            point_size=0.5,
            alpha=1.0,
            default_color='gray'
        )
        
        # Save the rendered image
        output_dir_path = os.path.join(output_path, output_dir)
        os.makedirs(output_dir_path, exist_ok=True)
        output_img_path = os.path.join(output_dir_path, 'error_visualization.png')
        
        # Handle RGBA images (if transparent_background=True was used)
        if rendered_img.shape[2] == 4:
            # Convert RGBA to RGB by compositing on white background
            alpha = rendered_img[:, :, 3:4] / 255.0
            rgb = rendered_img[:, :, :3]
            rendered_img = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
        
        # Create figure with colorbar
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rendered_img)
        ax.axis('off')
        
        # Get distance range for colorbar
        distance_min = distance_np.min()
        distance_max = distance_np.max()
        
        # Create colorbar using the same colormap (reuse colormap from above)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=distance_min, vmax=distance_max))
        sm.set_array([])
        
        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance Error (cm)', rotation=270, labelpad=20, fontsize=12)
        
        # Save figure with colorbar
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f'{output_dir}: {distance_pred2gt.mean():.4f} cm, {distance_gt2pred.mean():.4f} cm - Saved visualization to {output_img_path}')

        
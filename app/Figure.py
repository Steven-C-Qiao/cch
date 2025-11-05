import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt

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


def load_ply_colored(filepath):
    """Load PLY file with colors. Returns vertices and colors (None if no colors)."""
    if TRIMESH_AVAILABLE:
        # Try using trimesh first (better for binary PLY)
        mesh = trimesh.load(filepath)
        if hasattr(mesh, 'vertices'):
            verts = mesh.vertices
            # Check for vertex colors in visual
            colors = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors
                if vertex_colors is not None and len(vertex_colors) > 0:
                    # Check if colors are actually present (not all zeros or default)
                    if vertex_colors.shape[1] >= 3:
                        colors = vertex_colors[:, :3]  # RGB only, skip alpha
                        # If colors are all zeros or uniform, treat as no color
                        if np.all(colors == 0) or (np.std(colors) < 1e-6):
                            colors = None
            return verts, colors
    
    # Fallback to PyVista
    if PV_AVAILABLE:
        mesh = pv.read(filepath)
        verts = mesh.points
        colors = None
        # Check for color properties in point data
        if 'red' in mesh.point_data and 'green' in mesh.point_data and 'blue' in mesh.point_data:
            red = mesh.point_data['red']
            green = mesh.point_data['green']
            blue = mesh.point_data['blue']
            # Check if colors are actually present (not all zeros)
            if not (np.all(red == 0) and np.all(green == 0) and np.all(blue == 0)):
                colors = np.stack([red, green, blue], axis=1)
        elif 'RGB' in mesh.point_data:
            rgb = mesh.point_data['RGB']
            if rgb.shape[1] >= 3:
                colors = rgb[:, :3]
                # Check if colors are actually present
                if np.all(colors == 0):
                    colors = None
        return verts, colors
    
    return None, None


def load_obj_colored(filepath):
    """Load OBJ file with colors. Returns vertices and colors (None if no colors)."""
    if TRIMESH_AVAILABLE:
        # Try using trimesh first (better for OBJ files)
        mesh = trimesh.load(filepath)
        if hasattr(mesh, 'vertices'):
            verts = mesh.vertices
            # Check for vertex colors in visual
            colors = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors
                if vertex_colors is not None and len(vertex_colors) > 0:
                    # Check if colors are actually present (not all zeros or default)
                    if vertex_colors.shape[1] >= 3:
                        colors = vertex_colors[:, :3]  # RGB only, skip alpha
                        # If colors are all zeros or uniform, treat as no color
                        if np.all(colors == 0) or (np.std(colors) < 1e-6):
                            colors = None
            return verts, colors
    
    # Fallback to PyVista
    if PV_AVAILABLE:
        mesh = pv.read(filepath)
        verts = mesh.points
        colors = None
        # Check for color properties in point data
        if 'red' in mesh.point_data and 'green' in mesh.point_data and 'blue' in mesh.point_data:
            red = mesh.point_data['red']
            green = mesh.point_data['green']
            blue = mesh.point_data['blue']
            # Check if colors are actually present (not all zeros)
            if not (np.all(red == 0) and np.all(green == 0) and np.all(blue == 0)):
                colors = np.stack([red, green, blue], axis=1)
        elif 'RGB' in mesh.point_data:
            rgb = mesh.point_data['RGB']
            if rgb.shape[1] >= 3:
                colors = rgb[:, :3]
                # Check if colors are actually present
                if np.all(colors == 0):
                    colors = None
        return verts, colors
    
    return None, None


def render_pointcloud_pyvista(verts, colors=None, center=None, camera_pos=None, max_range=None, point_size=0.7, alpha=0.5, default_color='gray'):
    """Render point cloud to image using PyVista with y-axis up, elev=10, azim=20"""
    if not PV_AVAILABLE or len(verts) == 0:
        # Return blank image
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        return img
    
    # Create point cloud
    pcd = pv.PolyData(verts)
    
    # Create plotter with explicit offscreen mode
    plotter = pv.Plotter(off_screen=True, window_size=[400, 400])
    
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
    img = plotter.screenshot(None, return_img=True)
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
    distance = max_range * 3.5
    
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


def main():
    # Directory containing PLY files
    ply_dir = "/scratch/u5au/chexuan.u5au/from_u5aa/cch/vis/00134_take1"
    # Directory containing OBJ files
    obj_dir = "/scratch/u5au/chexuan.u5au/from_u5aa/cch/vis/up2you_00134_Take1"
    
    # Select some files to visualize (you can modify this to select different files)
    # Separate GT and Pred files - GT in first row, Pred in second row, OBJ in third row
    timesteps = ['000', '025', '050', '075', '100']
    
    gt_files = [(f"gt_vp_{ts}.ply", f"GT {ts}") for ts in timesteps]
    pred_files = [(f"pred_vp_{ts}.ply", f"Pred {ts}") for ts in timesteps]
    obj_files = [(f"pred_mesh_aligned_{ts}.obj", f"Mesh {ts}") for ts in timesteps]
    
    # Load GT point clouds (first row)
    all_verts = []
    gt_pointclouds = []
    
    for filename, label in gt_files:
        filepath = os.path.join(ply_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} does not exist, skipping...")
            continue
        
        verts, colors = load_ply_colored(filepath)
        if verts is not None:
            all_verts.append(verts)
            gt_pointclouds.append((verts, colors, label))
            print(f"Loaded {filename}: {len(verts)} points")
        else:
            print(f"Failed to load {filename}")
    
    # Load Pred point clouds (second row)
    pred_pointclouds = []
    
    for filename, label in pred_files:
        filepath = os.path.join(ply_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} does not exist, skipping...")
            continue
        
        verts, colors = load_ply_colored(filepath)
        if verts is not None:
            all_verts.append(verts)
            pred_pointclouds.append((verts, colors, label))
            print(f"Loaded {filename}: {len(verts)} points")
        else:
            print(f"Failed to load {filename}")
    
    # Load OBJ meshes (third row)
    obj_pointclouds = []
    
    for filename, label in obj_files:
        filepath = os.path.join(obj_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} does not exist, skipping...")
            continue
        
        verts, colors = load_obj_colored(filepath)
        if verts is not None:
            all_verts.append(verts)
            obj_pointclouds.append((verts, colors, label))
            print(f"Loaded {filename}: {len(verts)} vertices")
        else:
            print(f"Failed to load {filename}")
    
    if len(gt_pointclouds) == 0 and len(pred_pointclouds) == 0 and len(obj_pointclouds) == 0:
        print("No point clouds or meshes loaded!")
        return
    
    # Compute combined bounding box for consistent scaling
    all_verts_combined = np.vstack(all_verts)
    center, camera_pos, max_range, front_vec = get_view_params(all_verts_combined)
    
    # Create figure with 3 rows (GT, Pred, OBJ)
    num_cols = max(len(gt_pointclouds), len(pred_pointclouds), len(obj_pointclouds))
    if num_cols == 0:
        num_cols = 4
    num_rows = 3
    subfig_size = 4
    
    fig = plt.figure(figsize=(subfig_size * num_cols, subfig_size * num_rows))
    
    # Render GT point clouds (row 1)
    for idx, (verts, colors, label) in enumerate(gt_pointclouds):
        ax = fig.add_subplot(num_rows, num_cols, idx + 1)  # Row 1: idx + 1
        img = render_pointcloud_pyvista(
            verts, colors=colors, 
            center=center, camera_pos=camera_pos, 
            max_range=max_range,
            default_color='gray'
        )
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    
    # Render Pred point clouds (row 2)
    for idx, (verts, colors, label) in enumerate(pred_pointclouds):
        ax = fig.add_subplot(num_rows, num_cols, num_cols + idx + 1)  # Row 2: num_cols + idx + 1
        img = render_pointcloud_pyvista(
            verts, colors=colors, 
            center=center, camera_pos=camera_pos, 
            max_range=max_range,
            default_color='blue',
            alpha=1.
        )
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    
    # Render OBJ meshes (row 3)
    for idx, (verts, colors, label) in enumerate(obj_pointclouds):
        ax = fig.add_subplot(num_rows, num_cols, 2 * num_cols + idx + 1)  # Row 3: 2 * num_cols + idx + 1
        img = render_pointcloud_pyvista(
            verts, colors=colors, 
            center=center, camera_pos=camera_pos, 
            max_range=max_range,
            default_color='green'
        )
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    
    plt.tight_layout(pad=0.1)
    output_path = os.path.join("pointcloud_visualization_00134_take1.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()


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

# try:
#     import trimesh
#     TRIMESH_AVAILABLE = True
# except ImportError:
#     TRIMESH_AVAILABLE = False
#     print("Warning: trimesh not available")

TRIMESH_AVAILABLE = False


def load_ply_colored(filepath):
    """Load PLY file with optional faces and colors.
    Returns (vertices, faces, colors). Faces or colors can be None if unavailable.
    """
    if TRIMESH_AVAILABLE:
        # Try using trimesh first (better for binary PLY)
        mesh = trimesh.load(filepath, force='mesh')
        if hasattr(mesh, 'vertices'):
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0 else None
            # Check for vertex colors in visual
            colors = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors
                if vertex_colors is not None and len(vertex_colors) > 0:
                    if vertex_colors.shape[1] >= 3:
                        colors = vertex_colors[:, :3]
                        if np.all(colors == 0) or (np.std(colors) < 1e-6):
                            colors = None
            return verts, faces, colors
    
    # Fallback to PyVista
    if PV_AVAILABLE:
        mesh = pv.read(filepath)
        verts = np.asarray(mesh.points)
        # PyVista stores faces in a flat array [3, i0, i1, i2, 3, j0, j1, j2, ...]
        faces = None
        if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0:
            faces_np = np.asarray(mesh.faces)
            if faces_np.ndim == 1 and len(faces_np) % 4 == 0:
                faces = faces_np.reshape(-1, 4)[:, 1:4]
            elif faces_np.ndim == 2 and faces_np.shape[1] >= 3:
                faces = faces_np[:, :3]
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
        return verts, faces, colors
    
    return None, None, None


def load_obj_colored(filepath):
    """Load OBJ file with optional faces and colors.
    Returns (vertices, faces, colors). Faces or colors can be None if unavailable.
    """
    if TRIMESH_AVAILABLE:
        # Try using trimesh first (better for OBJ files)
        mesh = trimesh.load(filepath, force='mesh')
        if hasattr(mesh, 'vertices'):
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0 else None
            # Check for vertex colors in visual
            colors = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors
                if vertex_colors is not None and len(vertex_colors) > 0:
                    if vertex_colors.shape[1] >= 3:
                        colors = vertex_colors[:, :3]  # RGB only, skip alpha
                        if np.all(colors == 0) or (np.std(colors) < 1e-6):
                            colors = None
            return verts, faces, colors
    
    # Fallback to PyVista
    if PV_AVAILABLE:
        mesh = pv.read(filepath)
        verts = np.asarray(mesh.points)
        faces = None
        if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0:
            faces_np = np.asarray(mesh.faces)
            if faces_np.ndim == 1 and len(faces_np) % 4 == 0:
                faces = faces_np.reshape(-1, 4)[:, 1:4]
            elif faces_np.ndim == 2 and faces_np.shape[1] >= 3:
                faces = faces_np[:, :3]
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
        return verts, faces, colors
    
    return None, None, None


def render_pointcloud_pyvista(verts, colors=None, center=None, camera_pos=None, max_range=None, point_size=1., alpha=0.5, default_color='gray'):
    """Render point cloud to image using PyVista with y-axis up, elev=10, azim=20"""
    if not PV_AVAILABLE or len(verts) == 0:
        # Return blank image
        img = np.zeros((240, 240, 3), dtype=np.uint8)
        return img
    
    # Create point cloud
    pcd = pv.PolyData(verts)
    
    # Create plotter with explicit offscreen mode (portrait window, lower resolution)
    plotter = pv.Plotter(off_screen=True, window_size=[200, 300])
    
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
    plotter = pv.Plotter(off_screen=True, window_size=[200, 300])
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
    # Configuration: Select which rows to plot
    # Options: 'GT', 'UP2You', 'Ours'
    # rows_to_plot = ['GT', 'UP2You', 'Ours']  # Change this to select specific rows, e.g., ['GT', 'Ours'] or ['UP2You']
    rows_to_plot = ['Ours']
    output_path = os.path.join("Figures/vis_00134_take1_exp_090.png")
    
    # Directory containing PLY files
    ply_dir = "/scratch/u5au/chexuan.u5au/from_u5aa/cch/Figures/vis/00134_take1_exp_090"
    # Directory containing OBJ files
    obj_dir = "/scratch/u5au/chexuan.u5au/from_u5aa/cch/Figures/vis/up2you_00134_take1"
    
    # Select some files to visualize (you can modify this to select different files)
    # Layout: GT in first row, OBJ (from obj_dir) in second row, Pred in last row
    timesteps = ['000', '025', '050', '075', '100']
    
    # Load ALL available gt ply files dynamically
    try:
        gt_fnames = [f for f in os.listdir(ply_dir) if f.startswith('gt_vp_') and f.endswith('.ply')]
        def gt_sort_key(name):
            base = os.path.splitext(name)[0]
            token = base.split('_')[-1]
            try:
                return int(token)
            except ValueError:
                return token
        gt_fnames = sorted(gt_fnames, key=gt_sort_key)
    except Exception:
        gt_fnames = []
    gt_files = [(f, f"GT {os.path.splitext(f)[0].split('_')[-1]}") for f in gt_fnames]
    # Load ALL available pred ply files dynamically
    try:
        pred_fnames = [f for f in os.listdir(ply_dir) if f.startswith('pred_vp_') and f.endswith('.ply')]
        # Sort numerically by the trailing token if possible
        def pred_sort_key(name):
            base = os.path.splitext(name)[0]
            token = base.split('_')[-1]
            try:
                return int(token)
            except ValueError:
                return token
        pred_fnames = sorted(pred_fnames, key=pred_sort_key)
    except Exception:
        pred_fnames = []
    pred_files = [(f, f"Pred {os.path.splitext(f)[0].split('_')[-1]}") for f in pred_fnames]
    obj_files = [(f"pred_mesh_aligned_{ts}.obj", f"Mesh {ts}") for ts in timesteps]
    
    # Load data only for selected rows
    all_verts = []
    gt_geoms = []  # list of (verts, faces, colors, label)
    
    if 'GT' in rows_to_plot:
        for filename, label in gt_files:
            filepath = os.path.join(ply_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} does not exist, skipping...")
                continue
            
            verts, faces, colors = load_ply_colored(filepath)
            if verts is not None:
                all_verts.append(verts)
                gt_geoms.append((verts, faces, colors, label))
                print(f"Loaded {filename}: {len(verts)} points")
            else:
                print(f"Failed to load {filename}")
    
    # Load Pred point clouds
    pred_pointclouds = []
    
    if 'Ours' in rows_to_plot:
        for filename, label in pred_files:
            filepath = os.path.join(ply_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} does not exist, skipping...")
                continue
            
            verts, faces_dummy, colors = load_ply_colored(filepath)
            if verts is not None:
                all_verts.append(verts)
                pred_pointclouds.append((verts, colors, label))
                print(f"Loaded {filename}: {len(verts)} points")
            else:
                print(f"Failed to load {filename}")
    
    # Load OBJ meshes
    obj_geoms = []  # list of (verts, faces, colors, label)
    
    if 'UP2You' in rows_to_plot:
        for filename, label in obj_files:
            filepath = os.path.join(obj_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} does not exist, skipping...")
                continue
            
            verts, faces, colors = load_obj_colored(filepath)
            if verts is not None:
                all_verts.append(verts)
                obj_geoms.append((verts, faces, colors, label))
                print(f"Loaded {filename}: {len(verts)} vertices, faces: {0 if faces is None else len(faces)}")
            else:
                print(f"Failed to load {filename}")
    
    if len(gt_geoms) == 0 and len(pred_pointclouds) == 0 and len(obj_geoms) == 0:
        print("No point clouds or meshes loaded!")
        return
    
    # Compute combined bounding box for consistent scaling
    if len(all_verts) == 0:
        print("No vertices to plot!")
        return
    all_verts_combined = np.vstack(all_verts)
    center, camera_pos, max_range, front_vec = get_view_params(all_verts_combined)
    
    # Limit first and third rows to first 21 items for layout and rendering
    gt_geoms_limited = gt_geoms[:21] if 'GT' in rows_to_plot else []
    pred_pointclouds_limited = pred_pointclouds[:21] if 'Ours' in rows_to_plot else []
    
    # Map row names to their data and properties
    row_config = {
        'GT': {'data': gt_geoms_limited, 'label': 'GT', 'render_func': 'mesh', 'overlap': True, 'default_color': 'lightgray'},
        'UP2You': {'data': obj_geoms, 'label': 'UP2You', 'render_func': 'mesh', 'overlap': False, 'default_color': 'green'},
        'Ours': {'data': pred_pointclouds_limited, 'label': 'Ours', 'render_func': 'pointcloud', 'overlap': True, 'default_color': 'blue'}
    }
    
    # Create figure with selected rows
    num_cols = max(
        len(gt_geoms_limited) if 'GT' in rows_to_plot else 0,
        len(pred_pointclouds_limited) if 'Ours' in rows_to_plot else 0,
        len(obj_geoms) if 'UP2You' in rows_to_plot else 0
    )
    if num_cols == 0:
        num_cols = 4
    num_rows = len(rows_to_plot)
    subfig_size = 4
    
    # Portrait-oriented figure (taller than wide)
    fig = plt.figure(figsize=(subfig_size * num_cols, subfig_size * num_rows * 1.4))
    fig.patch.set_alpha(0)  # Make figure background transparent
    
    # Render rows based on selection
    row_axes_dict = {}  # Maps row name to list of axes
    for row_idx, row_name in enumerate(rows_to_plot):
        if row_name not in row_config:
            continue
        config = row_config[row_name]
        data = config['data']
        if len(data) == 0:
            row_axes_dict[row_name] = []
            continue
        
        axes_list = []
        for col_idx, item in enumerate(data):
            # Calculate subplot index based on row position
            subplot_idx = row_idx * num_cols + col_idx + 1
            ax = fig.add_subplot(num_rows, num_cols, subplot_idx)
            
            # Render based on type
            if config['render_func'] == 'mesh':
                verts, faces, colors = item[:3]  # gt_geoms format: (verts, faces, colors, label)
                img = render_mesh_pyvista(
                    verts, faces, colors=colors, center=center, camera_pos=camera_pos, 
                    max_range=max_range, opacity=1.0, default_color=config['default_color']
                )
            else:  # pointcloud
                verts, colors = item[0], item[1]  # pred_pointclouds format: (verts, colors, label)
                img = render_pointcloud_pyvista(
                    verts, colors=colors, center=center, camera_pos=camera_pos,
                    max_range=max_range, default_color=config['default_color'],
                    point_size=1.5, alpha=1.
                )
            
            ax.imshow(img)
            ax.axis('off')
            ax.patch.set_alpha(0)  # Make background transparent
            axes_list.append(ax)
        
        row_axes_dict[row_name] = axes_list
    
    # Extract axes lists for easier access
    first_row_axes = row_axes_dict.get('GT', [])
    second_row_axes = row_axes_dict.get('UP2You', [])
    third_row_axes = row_axes_dict.get('Ours', [])
    
    # Apply tight layout first, then adjust rows based on their overlap settings
    plt.tight_layout()
    try:
        overlap_fraction = 0.3
        # Apply overlap to rows that have overlap enabled
        for row_name in rows_to_plot:
            if row_name not in row_config:
                continue
            config = row_config[row_name]
            axes_list = row_axes_dict.get(row_name, [])
            
            if config['overlap'] and len(axes_list) > 1:
                # Apply overlap to this row
                pos0 = axes_list[0].get_position()
                width = pos0.width
                height = pos0.height
                y0 = pos0.y0
                x0 = pos0.x0
                step = width * (1.0 - overlap_fraction)
                for i, ax in enumerate(axes_list):
                    new_x0 = x0 + i * step
                    ax.set_position([new_x0, y0, width, height])
            elif not config['overlap'] and len(axes_list) > 0:
                # Distribute evenly across bounds from overlapping rows
                centers_left = []
                centers_right = []
                # Find overlapping rows for alignment
                for other_row_name in rows_to_plot:
                    if other_row_name == row_name:
                        continue
                    other_config = row_config.get(other_row_name)
                    if other_config and other_config['overlap']:
                        other_axes = row_axes_dict.get(other_row_name, [])
                        if len(other_axes) > 0:
                            pos_first = other_axes[0].get_position()
                            centers_left.append(pos_first.x0 + pos_first.width * 0.5)
                            pos_last = other_axes[-1].get_position()
                            centers_right.append(pos_last.x0 + pos_last.width * 0.5)
                
                # Reference height and y from current row
                pos_ref = axes_list[0].get_position()
                height = pos_ref.height
                y0 = pos_ref.y0
                
                if len(centers_left) > 0 and len(centers_right) > 0:
                    c_left = sum(centers_left) / len(centers_left)
                    c_right = sum(centers_right) / len(centers_right)
                else:
                    # Fallback: use current row span
                    pos0 = axes_list[0].get_position()
                    posN = axes_list[-1].get_position()
                    c_left = pos0.x0 + pos0.width * 0.5
                    c_right = posN.x0 + posN.width * 0.5
                
                # Compute centers for each subplot evenly between c_left and c_right
                n = len(axes_list)
                if n == 1:
                    target_centers = [c_left]
                else:
                    target_centers = [c_left + (c_right - c_left) * (i / (n - 1)) for i in range(n)]
                # Use each axis' current width to place centered
                for ax, cx in zip(axes_list, target_centers):
                    pos = ax.get_position()
                    w = pos.width
                    new_x0 = cx - 0.5 * w
                    ax.set_position([new_x0, y0, w, height])
    except Exception:
        print("Error in applying tight layout")
        pass
    # Add row labels at the top-left corner of each row with larger font
    try:
        label_fontsize = 20
        label_offset_x = 0.005
        label_offset_y = 0.01
        for row_name in rows_to_plot:
            axes_row = row_axes_dict.get(row_name, [])
            if len(axes_row) == 0:
                continue
            config = row_config.get(row_name, {})
            label_text = config.get('label', row_name)
            pos = axes_row[0].get_position()
            x = pos.x0 + label_offset_x
            y = pos.y1 - label_offset_y
            fig.text(x, y, label_text, va='top', ha='left', fontsize=label_fontsize, fontweight='bold')
    except Exception:
        pass
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', transparent=True)
    print(f"Saved visualization to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()


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
                        colors = vertex_colors[:, :3].copy()
                        # Colors from trimesh are typically uint8 [0, 255], convert to float for consistency
                        # But check if they're already in [0, 1] range
                        if colors.max() > 1.0:
                            # Colors are in [0, 255] range, keep as is (will be handled in renderer)
                            pass
                        # Check if colors are actually present (not all zeros or uniform)
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
        # Try multiple possible color formats
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
                colors = rgb[:, :3].copy()
                # Check if colors are actually present
                if np.all(colors == 0) or (np.std(colors) < 1e-6):
                    colors = None
        elif 'RGBA' in mesh.point_data:
            rgba = mesh.point_data['RGBA']
            if rgba.shape[1] >= 3:
                colors = rgba[:, :3].copy()
                # Check if colors are actually present
                if np.all(colors == 0) or (np.std(colors) < 1e-6):
                    colors = None
        # Check if active scalars contain RGB data (common for PLY files with vertex colors)
        if colors is None and mesh.point_data.keys():
            # Try to find any array with shape (N, 3) or (N, 4) that might be colors
            for key in mesh.point_data.keys():
                data = mesh.point_data[key]
                if data.ndim == 2 and (data.shape[1] == 3 or data.shape[1] == 4):
                    # Check if values are in reasonable color range [0, 255] or [0, 1]
                    if data.max() <= 255 and data.min() >= 0:
                        colors = data[:, :3].copy()
                        # Check if colors are actually present
                        if not (np.all(colors == 0) or (np.std(colors) < 1e-6)):
                            break
                        else:
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


def compute_vertex_normals(verts, faces):
    """Compute vertex normals by averaging adjacent face normals.
    Uses optimized functions from trimesh or PyVista if available.
    """
    if faces is None or len(faces) == 0:
        return None
    
    faces_np = np.asarray(faces, dtype=np.int64)
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        return None
    
    # Try trimesh first (fastest)
    if TRIMESH_AVAILABLE:
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces_np)
            vertex_normals = mesh.vertex_normals
            return vertex_normals
        except Exception:
            pass
    
    # Try PyVista (also fast)
    if PV_AVAILABLE:
        try:
            # Build PyVista face array format: [3, i0, i1, i2, 3, j0, j1, j2, ...]
            num_faces = faces_np.shape[0]
            faces_pv = np.hstack([np.full((num_faces, 1), 3, dtype=np.int64), faces_np]).reshape(-1)
            mesh = pv.PolyData(verts, faces_pv)
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
            vertex_normals = mesh['Normals']
            return vertex_normals
        except Exception:
            pass
    
    # Fallback: vectorized numpy implementation (faster than loop)
    verts_np = np.asarray(verts, dtype=np.float32)
    num_verts = len(verts_np)
    vertex_normals = np.zeros((num_verts, 3), dtype=np.float32)
    
    # Vectorized computation of face normals
    v0 = verts_np[faces_np[:, 0]]  # (N_faces, 3)
    v1 = verts_np[faces_np[:, 1]]  # (N_faces, 3)
    v2 = verts_np[faces_np[:, 2]]  # (N_faces, 3)
    
    edge1 = v1 - v0  # (N_faces, 3)
    edge2 = v2 - v0  # (N_faces, 3)
    face_normals = np.cross(edge1, edge2)  # (N_faces, 3)
    
    # Normalize face normals
    face_normals_len = np.linalg.norm(face_normals, axis=1, keepdims=True)  # (N_faces, 1)
    face_normals_len[face_normals_len < 1e-8] = 1.0  # Avoid division by zero
    face_normals = face_normals / face_normals_len  # (N_faces, 3)
    
    # Accumulate face normals to vertices using numpy advanced indexing
    np.add.at(vertex_normals, faces_np[:, 0], face_normals)
    np.add.at(vertex_normals, faces_np[:, 1], face_normals)
    np.add.at(vertex_normals, faces_np[:, 2], face_normals)
    
    # Normalize vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)  # (N_verts, 1)
    norms[norms < 1e-8] = 1.0  # Avoid division by zero
    vertex_normals = vertex_normals / norms  # (N_verts, 3)
    
    return vertex_normals


def normals_to_colors(normals):
    """Convert surface normals to RGB colors.
    Maps normal direction [-1, 1] to RGB [0, 1] by: RGB = (normal + 1) / 2
    """
    if normals is None:
        return None
    # Normalize to [0, 1] range: (normal + 1) / 2
    colors = (normals + 1.0) / 2.0
    # Clip to valid range
    colors = np.clip(colors, 0.0, 1.0)
    return colors


def render_pointcloud_pyvista(verts, colors=None, center=None, camera_pos=None, max_range=None, point_size=1., alpha=0.5, default_color='gray'):
    """Render point cloud to image using PyVista with y-axis up, elev=10, azim=20"""
    if not PV_AVAILABLE or len(verts) == 0:
        # Return blank image
        img = np.zeros((240, 240, 3), dtype=np.uint8)
        return img
    
    # Create point cloud
    pcd = pv.PolyData(verts)
    
    # Create plotter with explicit offscreen mode (portrait window, lower resolution)
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


def main(rows_to_plot, output_path, ply_dir, obj_dir, timesteps, poisson_dir=None, column_indices=None, gt_use_normals=False, ours_use_normals=True):
    # Note: poisson_dir is used for loading Ours meshes, but we also support point clouds
    # Configuration: Select which rows to plot
    # Options: 'GT', 'UP2You', 'Ours'
    # rows_to_plot = ['GT', 'UP2You', 'Ours']  # Change this to select specific rows, e.g., ['GT', 'Ours'] or ['UP2You']
    
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
    # Load ALL available pred_init ply files dynamically
    try:
        pred_init_fnames = [
            f for f in os.listdir(ply_dir)
            if f.startswith('pred_vp_init_') and f.endswith('.ply')
        ]
        def pred_init_sort_key(name):
            base = os.path.splitext(name)[0]
            token = base.split('_')[-1]
            try:
                return int(token)
            except ValueError:
                return token
        pred_init_fnames = sorted(pred_init_fnames, key=pred_init_sort_key)
    except Exception:
        pred_init_fnames = []
    pred_init_files = [(f, f"PredInit {os.path.splitext(f)[0].split('_')[-1]}") for f in pred_init_fnames]
    
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
                # Compute surface normals and convert to colors if requested
                if gt_use_normals and faces is not None and len(faces) > 0:
                    vertex_normals = compute_vertex_normals(verts, faces)
                    if vertex_normals is not None:
                        colors = normals_to_colors(vertex_normals)
                        print(f"Loaded {filename}: {len(verts)} vertices, {len(faces)} faces, using normal visualization")
                    else:
                        print(f"Warning: Could not compute normals for {filename}, using original colors")
                else:
                    print(f"Loaded {filename}: {len(verts)} points")
                
                all_verts.append(verts)
                gt_geoms.append((verts, faces, colors, label))
            else:
                print(f"Failed to load {filename}")
    
    # Load Ours meshes from poisson_reconstruction directory
    ours_geoms = []  # list of (verts, faces, colors, label)
    
    if 'Ours' in rows_to_plot:
        if poisson_dir is None:
            print("Warning: poisson_dir not provided, skipping Ours row")
        else:
            # Match GT files to find corresponding poisson meshes
            for filename, label in gt_files:
                # Extract timestep from GT filename (e.g., 'gt_vp_000.ply' -> '000')
                timestep = os.path.splitext(filename)[0].split('_')[-1]
                # Look for poisson mesh: pred_vp_{timestep}_poisson.obj
                poisson_filename = f"pred_vp_{timestep}_poisson.obj"
                filepath = os.path.join(poisson_dir, poisson_filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: {filepath} does not exist, skipping...")
                    continue
                
                verts, faces, colors = load_obj_colored(filepath)
                if verts is not None:
                    # Compute surface normals and convert to colors if requested
                    if ours_use_normals and faces is not None and len(faces) > 0:
                        vertex_normals = compute_vertex_normals(verts, faces)
                        if vertex_normals is not None:
                            colors = normals_to_colors(vertex_normals)
                            print(f"Loaded {poisson_filename}: {len(verts)} vertices, {len(faces)} faces, using normal visualization")
                        else:
                            print(f"Warning: Could not compute normals for {poisson_filename}, using default color")
                            colors = None
                    else:
                        if faces is None or len(faces) == 0:
                            print(f"Warning: No faces found for {poisson_filename}")
                        else:
                            print(f"Loaded {poisson_filename}: {len(verts)} vertices, {len(faces)} faces, using default color")
                        colors = None  # Use default color when normals are disabled
                    
                    all_verts.append(verts)
                    ours_geoms.append((verts, faces, colors, label))
                else:
                    print(f"Failed to load {poisson_filename}")
    
    # Load Ours Init meshes from poisson_reconstruction directory
    ours_init_geoms = []  # list of (verts, faces, colors, label)
    
    if 'ours_init' in rows_to_plot:
        if poisson_dir is None:
            print("Warning: poisson_dir not provided, skipping ours_init row")
        else:
            # Match GT files to find corresponding poisson meshes for init
            for filename, label in gt_files:
                # Extract timestep from GT filename (e.g., 'gt_vp_000.ply' -> '000')
                timestep = os.path.splitext(filename)[0].split('_')[-1]
                # Look for poisson mesh: pred_vp_init_{timestep}_poisson.obj
                poisson_filename = f"pred_vp_init_{timestep}_poisson.obj"
                filepath = os.path.join(poisson_dir, poisson_filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: {filepath} does not exist, skipping...")
                    continue
                
                verts, faces, colors = load_obj_colored(filepath)
                if verts is not None:
                    # Compute surface normals and convert to colors if requested
                    if ours_use_normals and faces is not None and len(faces) > 0:
                        vertex_normals = compute_vertex_normals(verts, faces)
                        if vertex_normals is not None:
                            colors = normals_to_colors(vertex_normals)
                            print(f"Loaded {poisson_filename}: {len(verts)} vertices, {len(faces)} faces, using normal visualization")
                        else:
                            print(f"Warning: Could not compute normals for {poisson_filename}, using default color")
                            colors = None
                    else:
                        if faces is None or len(faces) == 0:
                            print(f"Warning: No faces found for {poisson_filename}")
                        else:
                            print(f"Loaded {poisson_filename}: {len(verts)} vertices, {len(faces)} faces, using default color")
                        colors = None  # Use default color when normals are disabled
                    
                    all_verts.append(verts)
                    ours_init_geoms.append((verts, faces, colors, label))
                else:
                    print(f"Failed to load {poisson_filename}")
    
    # Load OBJ meshes from UP2You outputs_* directories
    obj_geoms = []  # list of (verts, faces, colors, label)
    
    if 'UP2You' in rows_to_plot:
        # Find all outputs_* directories in obj_dir
        try:
            all_items = os.listdir(obj_dir)
            output_dirs = [d for d in all_items if d.startswith('outputs_') and os.path.isdir(os.path.join(obj_dir, d))]
            
            # Sort numerically by the number after 'outputs_'
            def output_sort_key(name):
                try:
                    return int(name.replace('outputs_', ''))
                except ValueError:
                    return -1
            output_dirs = sorted(output_dirs, key=output_sort_key)
        except Exception:
            output_dirs = []
        
        # Match UP2You meshes to GT files by index
        for idx, gt_filename in enumerate(gt_fnames):
            if idx >= len(output_dirs):
                break
            
            output_dir = output_dirs[idx]
            mesh_path = os.path.join(obj_dir, output_dir, 'meshes', 'pred_mesh_aligned.obj')
            
            if not os.path.exists(mesh_path):
                print(f"Warning: {mesh_path} does not exist, skipping...")
                continue
            
            # Extract timestep from GT filename for label
            timestep = os.path.splitext(gt_filename)[0].split('_')[-1]
            label = f"UP2You {timestep}"
            
            verts, faces, colors = load_obj_colored(mesh_path)
            if verts is not None:
                # Compute surface normals and convert to colors
                if faces is not None and len(faces) > 0:
                    vertex_normals = compute_vertex_normals(verts, faces)
                    if vertex_normals is not None:
                        colors = normals_to_colors(vertex_normals)
                        print(f"Loaded {output_dir}/meshes/pred_mesh_aligned.obj: {len(verts)} vertices, {len(faces)} faces, using normal visualization")
                    else:
                        print(f"Warning: Could not compute normals for {output_dir}/meshes/pred_mesh_aligned.obj, using original colors")
                else:
                    print(f"Warning: No faces found for {output_dir}/meshes/pred_mesh_aligned.obj")
                
                all_verts.append(verts)
                obj_geoms.append((verts, faces, colors, label))
            else:
                print(f"Failed to load {mesh_path}")
    
    if len(gt_geoms) == 0 and len(ours_geoms) == 0 and len(ours_init_geoms) == 0 and len(obj_geoms) == 0:
        print("No point clouds or meshes loaded!")
        return
    
    # Compute combined bounding box for consistent scaling
    if len(all_verts) == 0:
        print("No vertices to plot!")
        return
    all_verts_combined = np.vstack(all_verts)
    center, camera_pos, max_range, front_vec = get_view_params(all_verts_combined)
    
    # Filter by column_indices if provided, otherwise use all items
    if column_indices is not None:
        # Cherry-pick specific indices
        gt_geoms_limited = [gt_geoms[i] for i in column_indices if i < len(gt_geoms)] if 'GT' in rows_to_plot else []
        ours_geoms_limited = [ours_geoms[i] for i in column_indices if i < len(ours_geoms)] if 'Ours' in rows_to_plot else []
        ours_init_geoms_limited = [ours_init_geoms[i] for i in column_indices if i < len(ours_init_geoms)] if 'ours_init' in rows_to_plot else []
        obj_geoms_limited = [obj_geoms[i] for i in column_indices if i < len(obj_geoms)] if 'UP2You' in rows_to_plot else []
    else:
        # Use all available items
        gt_geoms_limited = gt_geoms if 'GT' in rows_to_plot else []
        ours_geoms_limited = ours_geoms if 'Ours' in rows_to_plot else []
        ours_init_geoms_limited = ours_init_geoms if 'ours_init' in rows_to_plot else []
        obj_geoms_limited = obj_geoms if 'UP2You' in rows_to_plot else []
    
    # Map row names to their data and properties
    row_config = {
        'GT': {'data': gt_geoms_limited, 'label': 'Scan ground truth', 'render_func': 'mesh', 'overlap': True, 'default_color': 'lightgray'},
        'UP2You': {'data': obj_geoms_limited, 'label': 'UP2You', 'render_func': 'mesh', 'overlap': False, 'default_color': 'green'},
        'Ours': {'data': ours_geoms_limited, 'label': '', 'render_func': 'mesh', 'overlap': True, 'default_color': 'lightgray'},
        'ours_init': {'data': ours_init_geoms_limited, 'label': 'Initial reconstruction w/o blendshapes', 'render_func': 'mesh', 'overlap': True, 'default_color': 'lightgray'}
    }
    
    # Create figure with selected rows
    num_cols = max(
        len(gt_geoms_limited) if 'GT' in rows_to_plot else 0,
        len(ours_geoms_limited) if 'Ours' in rows_to_plot else 0,
        len(obj_geoms_limited) if 'UP2You' in rows_to_plot else 0,
        len(ours_init_geoms_limited) if 'ours_init' in rows_to_plot else 0
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
                verts, faces, colors = item[:3]  # gt_geoms/ours_geoms format: (verts, faces, colors, label)
                img = render_mesh_pyvista(
                    verts, faces, colors=colors, center=center, camera_pos=camera_pos, 
                    max_range=max_range, opacity=1.0, default_color=config['default_color']
                )
            else:  # pointcloud
                verts, colors = item[0], item[1]  # pred_pointclouds format: (verts, colors, label)
                img = render_pointcloud_pyvista(
                    verts, colors=colors, center=center, camera_pos=camera_pos,
                    max_range=max_range, default_color=config['default_color'],
                    point_size=1.2, alpha=1.
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

    # rows_to_plot = ['GT', 'ours_init', 'Ours']
    # output_path = os.path.join("Figures/vis_00134_take1_exp_102.png")
    
    # # Directory containing PLY files
    # ply_dir = "/scratch/u5au/chexuan.u5au/from_u5aa/cch/Figures/vis/00134_take1_exp_102"
    # # Directory containing OBJ files
    # obj_dir = "/scratch/u5au/chexuan.u5au/from_u5aa/cch/Figures/vis/up2you_00134_take1"



    # for take in ['Take1']:#, 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take9']:
    #     output_path = os.path.join(f"Figures/vis_00134_{take}_exp_104_ep9_ours_mesh.png")
    #     ply_dir = f"/scratch/u5au/chexuan.u5au/from_u5aa/cch/Figures/vis/00134_{take}_exp_104_ep9"
    #     # UP2You directory uses lowercase 'take1' instead of 'Take1'
    #     take_lower = take.lower()
    #     obj_dir = f"/scratch/u5au/chexuan.u5au/from_u5aa/cch/Figures/up2you_vis/00134_{take_lower}"
    #     poisson_dir = f"/scratch/u5au/chexuan.u5au/from_u5aa/cch/Figures/poisson_reconstruction/00134_{take}_exp_104_ep9"
        
    #     timesteps = ['000', '025', '050', '075', '100']
        
    #     main(rows_to_plot, output_path, ply_dir, obj_dir, timesteps, poisson_dir=poisson_dir)


    # for take in ['Take2']:#, 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take9']:
    #     output_path = os.path.join(f"Figures/vis/00191/00191_{take}_exp_100_2_ours_mesh.png")
    #     ply_dir = f"Figures/vis/00191/00191_{take}_exp_100_2"
        
    #     take_lower = take.lower()
    #     obj_dir = f"Figures/vis/up2you_vis/00191_{take_lower}"
    #     poisson_dir = f"Figures/vis/00191/00191_{take}_exp_100_2_poisson"
        
    #     timesteps = ['000', '025', '050', '075', '100']
        
    #     main(rows_to_plot, output_path, ply_dir, obj_dir, timesteps, poisson_dir=poisson_dir)


    '''----------------------------------------------------------
    A bunch of random poses 4DDress eval set
    ----------------------------------------------------------'''
    # rows_to_plot = ['Ours']
    # output_path = 'Figures/fig1_meshes.jpg'
    # ply_dir = f"exp/exp_100_5_vp/vis/for_visuals"
    
    # obj_dir = f"exp/exp_100_5_vp/vis/up2you_vis"
    # poisson_dir = f"exp/exp_100_5_vp/vis/poisson_reconstruction"
    
    # timesteps = ['000', '025', '050', '075', '100']
    
    # # Cherry-pick specific column indices (0-indexed)
    # # Example: column_indices = [0, 5, 10, 15, 20] to plot columns 0, 5, 10, 15, 20
    # # Set to None to plot all available columns (up to 21)
    # # column_indices = [1, 4, 5, 6, 7, 9, 11, 13, 17, 20]  # Change to e.g., [0, 5, 10, 15, 20] to cherry-pick
    # column_indices = [1, 5, 6, 7, 9, 10, 12, 13, 14, 17, 22, 23]
    
    # # Option to use surface normals for GT visualization (default: False, uses original colors if available)
    # gt_use_normals = False  # Set to True to visualize GT with surface normals
    # # Option to use surface normals for Ours visualization (default: True, uses normal visualization)
    # ours_use_normals = False  # Set to False to use original colors if available
    
    # main(rows_to_plot, output_path, ply_dir, obj_dir, timesteps, poisson_dir=poisson_dir, column_indices=column_indices, gt_use_normals=gt_use_normals, ours_use_normals=ours_use_normals)


    '''----------------------------------------------------------
    vs
    ----------------------------------------------------------'''
    rows_to_plot = ['GT', 'UP2You', 'Ours']

    output_path = 'Figures/vs_up2you//exp_single/for_visuals/mesh_vis_single_pose.jpg'
    ply_dir = f"Figures/vs_up2you/exp_single/for_visuals"
    poisson_dir = f"Figures/vs_up2you/exp_single/for_visuals/poisson_reconstruction"
    
    up2you_obj_dir = 'Figures/vs_up2you/exp_single/Take1_00079'
    
    timesteps = ['000']
    

    # Option to use surface normals for GT visualization (default: False, uses original colors if available)
    gt_use_normals = False  # Set to True to visualize GT with surface normals
    
    main(rows_to_plot, output_path, ply_dir, up2you_obj_dir, timesteps, poisson_dir=poisson_dir, gt_use_normals=gt_use_normals)

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
    """Render point cloud to image using PyVista with automatic camera positioning.
    Uses PyVista's automatic view range and distance calculation.
    """
    if not PV_AVAILABLE or len(verts) == 0:
        # Return blank image
        img = np.zeros((240, 240, 3), dtype=np.uint8)
        return img
    
    # Create point cloud
    pcd = pv.PolyData(verts)
    
    # Create plotter with explicit offscreen mode (portrait window, lower resolution)
    plotter = pv.Plotter(off_screen=True, window_size=[600, 600])
    
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
    
    # Use provided camera parameters if available, otherwise use automatic positioning
    if center is not None and camera_pos is not None:
        # Use provided camera parameters (for column alignment)
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]  # y-axis up
        plotter.camera.view_angle = 30.0  # Fixed field of view angle
    else:
        # Use PyVista's automatic camera positioning
        plotter.reset_camera()
        
        # Set y-axis up and desired viewing angle (elev=10, azim=20)
        # Get the center from the mesh bounds
        mesh_center = pcd.center
        bounds = pcd.bounds
        # Calculate distance from bounds
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        distance = max_dim * 2.0  # Distance multiplier for good view
        
        # Calculate camera position with elev=10, azim=20
        elev, azim = 10, 20
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        
        camera_offset = np.array([
            distance * np.cos(elev_rad) * np.sin(azim_rad),
            distance * np.sin(elev_rad),
            distance * np.cos(elev_rad) * np.cos(azim_rad)
        ])
        camera_pos = mesh_center + camera_offset
        
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = mesh_center
        plotter.camera.up = [0, 1, 0]  # y-axis up
        plotter.camera.view_angle = 30.0  # Fixed field of view angle
    
    # Render to numpy array
    # Capture with transparent background (returns RGBA)
    img = plotter.screenshot(None, return_img=True, transparent_background=True)
    plotter.close()
    
    return img


def render_mesh_pyvista(verts, faces, colors=None, center=None, camera_pos=None, max_range=None, opacity=1.0, default_color='lightgray'):
    """Render triangle mesh to image using PyVista with automatic camera positioning.
    Uses PyVista's automatic view range and distance calculation.
    """
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
    
    # Use provided camera parameters if available, otherwise use automatic positioning
    if center is not None and camera_pos is not None:
        # Use provided camera parameters (for column alignment)
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]  # y-axis up
        plotter.camera.view_angle = 30.0  # Fixed field of view angle
    else:
        # Use PyVista's automatic camera positioning
        plotter.reset_camera()
        
        # Set y-axis up and desired viewing angle (elev=10, azim=20)
        # Get the center from the mesh bounds
        mesh_center = mesh.center
        bounds = mesh.bounds
        # Calculate distance from bounds
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        distance = max_dim * 2.0  # Distance multiplier for good view
        
        # Calculate camera position with elev=10, azim=20
        elev, azim = 10, 20
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        
        camera_offset = np.array([
            distance * np.cos(elev_rad) * np.sin(azim_rad),
            distance * np.sin(elev_rad),
            distance * np.cos(elev_rad) * np.cos(azim_rad)
        ])
        camera_pos = mesh_center + camera_offset
        
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = mesh_center
        plotter.camera.up = [0, 1, 0]  # y-axis up
        plotter.camera.view_angle = 30.0  # Fixed field of view angle
    
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


def main(rows_to_plot, output_path, base_dir, column_indices=None, gt_use_normals=False):
    """
    Main function to visualize meshes from Take* subdirectories.
    
    Args:
        rows_to_plot: List of rows to plot, e.g., ['GT', 'UP2You', 'Ours']
        output_path: Path to save the output figure
        base_dir: Base directory containing Take* subdirectories
        column_indices: Optional list of indices to cherry-pick specific columns
        gt_use_normals: Whether to use surface normals for GT visualization
    """
    # Find all Take* subdirectories
    try:
        all_items = os.listdir(base_dir)
        take_dirs = [d for d in all_items if d.startswith('Take') and os.path.isdir(os.path.join(base_dir, d))]
        
        # Sort Take directories by name
        def take_sort_key(name):
            # Extract number after 'Take' for sorting (e.g., 'Take1_00079' -> 1)
            try:
                parts = name.replace('Take', '').split('_')
                return int(parts[0]) if parts[0].isdigit() else name
            except (ValueError, IndexError):
                return name
        take_dirs = sorted(take_dirs, key=take_sort_key)
    except Exception as e:
        print(f"Error listing directory {base_dir}: {e}")
        take_dirs = []
    
    if len(take_dirs) == 0:
        print(f"No Take* subdirectories found in {base_dir}")
        return
    
    # Load data only for selected rows
    all_verts = []
    gt_geoms = []  # list of (verts, faces, colors, label)
    ours_geoms = []  # list of (verts, faces, colors, label)
    obj_geoms = []  # list of (verts, faces, colors, label)
    
    # Process each Take* directory
    for take_dir in take_dirs:
        take_path = os.path.join(base_dir, take_dir)
        
        # Extract label from directory name (e.g., 'Take1_00079' -> 'Take1_00079')
        label_base = take_dir
        
        # Load GT mesh (gt.ply or gt.obj)
        if 'GT' in rows_to_plot:
            gt_filepath_ply = os.path.join(take_path, 'gt.ply')
            gt_filepath_obj = os.path.join(take_path, 'gt.obj')
            gt_filepath = None
            gt_filename = None
            
            # Check for gt.ply first, then gt.obj
            if os.path.exists(gt_filepath_ply):
                gt_filepath = gt_filepath_ply
                gt_filename = 'gt.ply'
            elif os.path.exists(gt_filepath_obj):
                gt_filepath = gt_filepath_obj
                gt_filename = 'gt.obj'
            
            if gt_filepath is not None:
                # Load based on file extension
                if gt_filename.endswith('.ply'):
                    verts, faces, colors = load_ply_colored(gt_filepath)
                else:  # .obj
                    verts, faces, colors = load_obj_colored(gt_filepath)
                
                if verts is not None:
                    # Compute surface normals and convert to colors if requested
                    if gt_use_normals and faces is not None and len(faces) > 0:
                        vertex_normals = compute_vertex_normals(verts, faces)
                        if vertex_normals is not None:
                            colors = normals_to_colors(vertex_normals)
                            print(f"Loaded {take_dir}/{gt_filename}: {len(verts)} vertices, {len(faces)} faces, using normal visualization")
                        else:
                            print(f"Warning: Could not compute normals for {take_dir}/{gt_filename}, using original colors")
                    else:
                        print(f"Loaded {take_dir}/{gt_filename}: {len(verts)} points")
                    
                    all_verts.append(verts)
                    gt_geoms.append((verts, faces, colors, f"GT {label_base}"))
                else:
                    print(f"Failed to load {take_dir}/{gt_filename}")
            else:
                print(f"Warning: Neither gt.ply nor gt.obj exists in {take_dir}, skipping GT")
        
        # Load Ours mesh (ours.obj)
        if 'Ours' in rows_to_plot:
            ours_filepath = os.path.join(take_path, 'ours.obj')
            if os.path.exists(ours_filepath):
                verts, faces, colors = load_obj_colored(ours_filepath)
                if verts is not None:
                    # Compute surface normals and convert to colors
                    if faces is not None and len(faces) > 0:
                        vertex_normals = compute_vertex_normals(verts, faces)
                        if vertex_normals is not None:
                            colors = normals_to_colors(vertex_normals)
                            print(f"Loaded {take_dir}/ours.obj: {len(verts)} vertices, {len(faces)} faces, using normal visualization")
                        else:
                            print(f"Warning: Could not compute normals for {take_dir}/ours.obj, using original colors")
                    else:
                        print(f"Warning: No faces found for {take_dir}/ours.obj")
                    
                    all_verts.append(verts)
                    ours_geoms.append((verts, faces, colors, f"Ours {label_base}"))
                else:
                    print(f"Failed to load {take_dir}/ours.obj")
            else:
                print(f"Warning: {ours_filepath} does not exist, skipping Ours for {take_dir}")
        
        # Load UP2You mesh (up2you.obj)
        if 'UP2You' in rows_to_plot:
            up2you_filepath = os.path.join(take_path, 'up2you.obj')
            if os.path.exists(up2you_filepath):
                verts, faces, colors = load_obj_colored(up2you_filepath)
                if verts is not None:
                    # Compute surface normals and convert to colors
                    if faces is not None and len(faces) > 0:
                        vertex_normals = compute_vertex_normals(verts, faces)
                        if vertex_normals is not None:
                            colors = normals_to_colors(vertex_normals)
                            print(f"Loaded {take_dir}/up2you.obj: {len(verts)} vertices, {len(faces)} faces, using normal visualization")
                        else:
                            print(f"Warning: Could not compute normals for {take_dir}/up2you.obj, using original colors")
                    else:
                        print(f"Warning: No faces found for {take_dir}/up2you.obj")
                    
                    all_verts.append(verts)
                    obj_geoms.append((verts, faces, colors, f"UP2You {label_base}"))
                else:
                    print(f"Failed to load {take_dir}/up2you.obj")
            else:
                print(f"Warning: {up2you_filepath} does not exist, skipping UP2You for {take_dir}")
    
    # Initialize ours_init_geoms as empty (not used in new structure)
    ours_init_geoms = []
    
    if len(gt_geoms) == 0 and len(ours_geoms) == 0 and len(ours_init_geoms) == 0 and len(obj_geoms) == 0:
        print("No point clouds or meshes loaded!")
        return
    
    # Filter by column_indices if provided, otherwise limit to first 21 items
    if column_indices is not None:
        # Cherry-pick specific indices
        gt_geoms_limited = [gt_geoms[i] for i in column_indices if i < len(gt_geoms)] if 'GT' in rows_to_plot else []
        ours_geoms_limited = [ours_geoms[i] for i in column_indices if i < len(ours_geoms)] if 'Ours' in rows_to_plot else []
        ours_init_geoms_limited = [ours_init_geoms[i] for i in column_indices if i < len(ours_init_geoms)] if 'ours_init' in rows_to_plot else []
        obj_geoms_limited = [obj_geoms[i] for i in column_indices if i < len(obj_geoms)] if 'UP2You' in rows_to_plot else []
    else:
        # Default: limit to first 21 items for layout and rendering
        gt_geoms_limited = gt_geoms[:21] if 'GT' in rows_to_plot else []
        ours_geoms_limited = ours_geoms[:21] if 'Ours' in rows_to_plot else []
        ours_init_geoms_limited = ours_init_geoms[:21] if 'ours_init' in rows_to_plot else []
        obj_geoms_limited = obj_geoms[:21] if 'UP2You' in rows_to_plot else []
    
    # Map row names to their data and properties
    row_config = {
        'GT': {'data': gt_geoms_limited, 'label': 'Scan', 'render_func': 'mesh', 'overlap': True, 'default_color': 'lightgray'},
        'UP2You': {'data': obj_geoms_limited, 'label': 'UP2You', 'render_func': 'mesh', 'overlap': False, 'default_color': 'green'},
        'Ours': {'data': ours_geoms_limited, 'label': 'Ours', 'render_func': 'mesh', 'overlap': True, 'default_color': 'blue'},
        'ours_init': {'data': ours_init_geoms_limited, 'label': 'Initial reconstruction w/o blendshapes', 'render_func': 'mesh', 'overlap': True, 'default_color': 'blue'}
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
            # Each mesh now uses automatic camera positioning based on its own bounds
            if config['render_func'] == 'mesh':
                verts, faces, colors = item[:3]  # gt_geoms/ours_geoms format: (verts, faces, colors, label)
                img = render_mesh_pyvista(
                    verts, faces, colors=colors, opacity=1.0, default_color=config['default_color']
                )
            else:  # pointcloud
                verts, colors = item[0], item[1]  # pred_pointclouds format: (verts, colors, label)
                img = render_pointcloud_pyvista(
                    verts, colors=colors, default_color=config['default_color'],
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
        overlap_fraction = 0.0  # Reduced from 0.3 to space out columns more
        column_gap = 0.1  # Additional gap between columns (as fraction of width)
        
        # Step 1: Apply overlap to rows that have overlap enabled
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
                step = width * (1.0 - overlap_fraction + column_gap)  # Add gap between columns
                for i, ax in enumerate(axes_list):
                    new_x0 = x0 + i * step
                    ax.set_position([new_x0, y0, width, height])
        
        # Step 2: Compute column centers from overlapping rows (after overlap is applied)
        # Collect all overlapping rows and compute column centers
        overlapping_row_axes = []
        for row_name in rows_to_plot:
            if row_name not in row_config:
                continue
            config = row_config[row_name]
            if config['overlap']:
                axes_list = row_axes_dict.get(row_name, [])
                if len(axes_list) > 0:
                    overlapping_row_axes.append(axes_list)
        
        # Compute column centers from overlapping rows
        column_centers = []
        if len(overlapping_row_axes) > 0:
            # Use the first overlapping row as reference for number of columns
            ref_axes = overlapping_row_axes[0]
            num_cols_actual = len(ref_axes)
            
            for col_idx in range(num_cols_actual):
                # Collect centers from all overlapping rows for this column
                col_centers = []
                for axes_list in overlapping_row_axes:
                    if col_idx < len(axes_list):
                        pos = axes_list[col_idx].get_position()
                        col_centers.append(pos.x0 + pos.width * 0.5)
                
                if len(col_centers) > 0:
                    # Average the centers from all overlapping rows
                    column_centers.append(sum(col_centers) / len(col_centers))
                else:
                    # Fallback
                    if col_idx < len(ref_axes):
                        pos = ref_axes[col_idx].get_position()
                        column_centers.append(pos.x0 + pos.width * 0.5)
        
        # Step 3: Align non-overlapping rows to the column centers
        for row_name in rows_to_plot:
            if row_name not in row_config:
                continue
            config = row_config[row_name]
            axes_list = row_axes_dict.get(row_name, [])
            
            if not config['overlap'] and len(axes_list) > 0 and len(column_centers) > 0:
                # Align this row's subplots to the column centers
                pos_ref = axes_list[0].get_position()
                height = pos_ref.height
                y0 = pos_ref.y0
                
                # Match each subplot to its corresponding column center
                for col_idx, ax in enumerate(axes_list):
                    if col_idx < len(column_centers):
                        cx = column_centers[col_idx]
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
    rows_to_plot = ['GT', 'UP2You', 'Ours']
    
    base_dir = 'Figures/vs_up2you'
    output_path = os.path.join(base_dir, 'comparison.png')
    
    # Option to use surface normals for GT visualization (default: False, uses original colors if available)
    gt_use_normals = False  # Set to True to visualize GT with surface normals
    
    # Optional: specify column indices to cherry-pick specific columns (e.g., [0, 1, 2])
    column_indices = None  # Set to None to use all available columns
    
    main(rows_to_plot, output_path, base_dir, column_indices=column_indices, gt_use_normals=gt_use_normals)


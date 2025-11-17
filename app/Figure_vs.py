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

def load_ply_colored(filepath):
    """Load PLY file with optional faces and colors.
    Returns (vertices, faces, colors). Faces or colors can be None if unavailable.
    """
    if TRIMESH_AVAILABLE:
        mesh = trimesh.load(filepath, force='mesh')
        if hasattr(mesh, 'vertices'):
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None and len(mesh.faces) > 0 else None
            colors = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors
                if vertex_colors is not None and len(vertex_colors) > 0:
                    if vertex_colors.shape[1] >= 3:
                        colors = vertex_colors[:, :3].copy()
                        if colors.max() > 1.0:
                            pass
                        if np.all(colors == 0) or (np.std(colors) < 1e-6):
                            colors = None
            return verts, faces, colors
    
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
        if 'red' in mesh.point_data and 'green' in mesh.point_data and 'blue' in mesh.point_data:
            red = mesh.point_data['red']
            green = mesh.point_data['green']
            blue = mesh.point_data['blue']
            if not (np.all(red == 0) and np.all(green == 0) and np.all(blue == 0)):
                colors = np.stack([red, green, blue], axis=1)
        elif 'RGB' in mesh.point_data:
            rgb = mesh.point_data['RGB']
            if rgb.shape[1] >= 3:
                colors = rgb[:, :3].copy()
                if np.all(colors == 0) or (np.std(colors) < 1e-6):
                    colors = None
        return verts, faces, colors
    
    return None, None, None

def compute_vertex_normals(verts, faces):
    """Compute vertex normals by averaging adjacent face normals."""
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
            num_faces = faces_np.shape[0]
            faces_pv = np.hstack([np.full((num_faces, 1), 3, dtype=np.int64), faces_np]).reshape(-1)
            mesh = pv.PolyData(verts, faces_pv)
            mesh = mesh.compute_normals(point_normals=True, cell_normals=False)
            vertex_normals = mesh['Normals']
            return vertex_normals
        except Exception:
            pass
    
    # Fallback: vectorized numpy implementation
    verts_np = np.asarray(verts, dtype=np.float32)
    num_verts = len(verts_np)
    vertex_normals = np.zeros((num_verts, 3), dtype=np.float32)
    
    v0 = verts_np[faces_np[:, 0]]
    v1 = verts_np[faces_np[:, 1]]
    v2 = verts_np[faces_np[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    
    face_normals_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals_len[face_normals_len < 1e-8] = 1.0
    face_normals = face_normals / face_normals_len
    
    np.add.at(vertex_normals, faces_np[:, 0], face_normals)
    np.add.at(vertex_normals, faces_np[:, 1], face_normals)
    np.add.at(vertex_normals, faces_np[:, 2], face_normals)
    
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    vertex_normals = vertex_normals / norms
    
    return vertex_normals

def normals_to_colors(normals):
    """Convert surface normals to RGB colors."""
    if normals is None:
        return None
    colors = (normals + 1.0) / 2.0
    colors = np.clip(colors, 0.0, 1.0)
    return colors

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

def get_view_params_rotated(x, rotation_degrees=180):
    """Get view parameters rotated by rotation_degrees around y-axis (elev=10, azim=20+rotation)"""
    max_range = np.array([
        x[:, 0].max() - x[:, 0].min(),
        x[:, 1].max() - x[:, 1].min(),
        x[:, 2].max() - x[:, 2].min()
    ]).max() / 2.0 + 0.1
    mid_x = (x[:, 0].max() + x[:, 0].min()) * 0.5
    mid_y = (x[:, 1].max() + x[:, 1].min()) * 0.5
    mid_z = (x[:, 2].max() + x[:, 2].min()) * 0.5
    center = [mid_x, mid_y, mid_z]
    
    elev, azim = 10, 20 + rotation_degrees
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

def render_mesh_pyvista(verts, faces, colors=None, center=None, camera_pos=None, max_range=None, opacity=1.0, default_color='lightgray', canvas_size=(1200, 800)):
    """Render triangle mesh to image using PyVista with automatic camera positioning."""
    if not PV_AVAILABLE or len(verts) == 0:
        return np.zeros((canvas_size[0], canvas_size[1], 4), dtype=np.uint8)
    
    if faces is None or len(faces) == 0:
        # Fallback to point rendering if mesh not available
        pcd = pv.PolyData(verts)
        plotter = pv.Plotter(off_screen=True, window_size=[canvas_size[1], canvas_size[0]])
        if colors is not None and len(colors) > 0:
            if colors.max() <= 1.0:
                colors_uint8 = (colors * 255).astype(np.uint8)
            else:
                colors_uint8 = colors.astype(np.uint8)
            pcd['colors'] = colors_uint8
            plotter.add_mesh(pcd, point_size=2.0, scalars='colors', rgb=True, opacity=opacity)
        else:
            plotter.add_mesh(pcd, point_size=2.0, color=default_color, opacity=opacity)
        
        if center is not None and camera_pos is not None:
            plotter.camera.position = camera_pos
            plotter.camera.focal_point = center
            plotter.camera.up = [0, 1, 0]
            plotter.camera.view_angle = 30.0
        
        img = plotter.screenshot(None, return_img=True, transparent_background=True)
        plotter.close()
        return img
    
    # Build PolyData from verts and triangular faces
    faces_tri = np.asarray(faces, dtype=np.int64)
    if faces_tri.ndim != 2 or faces_tri.shape[1] != 3:
        try:
            faces_tri = faces_tri[:, :3]
        except Exception:
            # Fallback to point rendering
            pcd = pv.PolyData(verts)
            plotter = pv.Plotter(off_screen=True, window_size=[canvas_size[1], canvas_size[0]])
            if colors is not None and len(colors) > 0:
                if colors.max() <= 1.0:
                    colors_uint8 = (colors * 255).astype(np.uint8)
                else:
                    colors_uint8 = colors.astype(np.uint8)
                pcd['colors'] = colors_uint8
                plotter.add_mesh(pcd, point_size=2.0, scalars='colors', rgb=True, opacity=opacity)
            else:
                plotter.add_mesh(pcd, point_size=2.0, color=default_color, opacity=opacity)
            
            if center is not None and camera_pos is not None:
                plotter.camera.position = camera_pos
                plotter.camera.focal_point = center
                plotter.camera.up = [0, 1, 0]
                plotter.camera.view_angle = 30.0
            
            img = plotter.screenshot(None, return_img=True, transparent_background=True)
            plotter.close()
            return img
    
    # PyVista face array format: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    num_faces = faces_tri.shape[0]
    faces_pv = np.hstack([np.full((num_faces, 1), 3, dtype=np.int64), faces_tri]).reshape(-1)
    mesh = pv.PolyData(verts, faces_pv)
    plotter = pv.Plotter(off_screen=True, window_size=[canvas_size[1], canvas_size[0]])
    
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
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = center
        plotter.camera.up = [0, 1, 0]
        plotter.camera.view_angle = 30.0
    else:
        # Use PyVista's automatic camera positioning
        plotter.reset_camera()
        mesh_center = mesh.center
        bounds = mesh.bounds
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        distance = max_dim * 2.0
        
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
        plotter.camera.up = [0, 1, 0]
        plotter.camera.view_angle = 30.0
    
    # Capture with transparent background (returns RGBA)
    img = plotter.screenshot(None, return_img=True, transparent_background=True)
    plotter.close()
    return img

def compute_all_distances(base_dir, stride=1, device='cuda', rows_to_plot=['GT', 'UP2You', 'Ours']):
    """Compute chamfer distances for all pairs from Take*/scaled directories and return global min/max."""
    all_distances = []
    up2you_data = []
    ours_data = []
    gt_data = []
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Find all Take* subdirectories
    try:
        all_items = os.listdir(base_dir)
        take_dirs = [d for d in all_items if d.startswith('Take') and os.path.isdir(os.path.join(base_dir, d))]
        
        # Sort Take directories by name
        def take_sort_key(name):
            try:
                parts = name.replace('Take', '').split('_')
                if parts[0].isdigit():
                    return (0, int(parts[0]))  # Numeric: sort by number
                else:
                    return (1, name)  # Non-numeric: sort alphabetically after numeric
            except (ValueError, IndexError):
                return (1, name)
        take_dirs = sorted(take_dirs, key=take_sort_key)
    except Exception as e:
        print(f"Error listing directory {base_dir}: {e}")
        take_dirs = []
    
    if len(take_dirs) == 0:
        print(f"No Take* subdirectories found in {base_dir}")
        return up2you_data, ours_data, gt_data, 0.0, 1.0
    
    # Process each Take* directory
    for take_dir in take_dirs[::stride]:
        take_path = os.path.join(base_dir, take_dir)
        scaled_path = os.path.join(take_path, 'scaled')
        
        if not os.path.exists(scaled_path):
            print(f"Warning: scaled directory not found in {take_dir}, skipping error computation")
            continue
        
        label_base = take_dir
        
        # Load GT mesh from scaled directory for error computation
        if 'GT' in rows_to_plot:
            gt_filepath = os.path.join(scaled_path, 'gt_scaled.obj')
            if os.path.exists(gt_filepath):
                try:
                    gt_verts, gt_faces, _ = load_obj_colored(gt_filepath)
                    if gt_verts is not None:
                        gt_data.append({
                            'gt_vertices': gt_verts,
                            'gt_faces': gt_faces,
                            'label': label_base
                        })
                except Exception as e:
                    print(f'Error loading GT {take_dir}: {e}')
                    gt_data.append(None)
            else:
                gt_data.append(None)
        
        # Load UP2You mesh from scaled directory for error computation
        if 'UP2You' in rows_to_plot:
            up2you_filepath = os.path.join(scaled_path, 'up2you_scaled.obj')
            gt_filepath = os.path.join(scaled_path, 'gt_scaled.obj')
            
            if os.path.exists(up2you_filepath) and os.path.exists(gt_filepath):
                try:
                    pred_verts, pred_faces, _ = load_obj_colored(up2you_filepath)
                    gt_verts, gt_faces, _ = load_obj_colored(gt_filepath)
                    
                    if pred_verts is not None and gt_verts is not None:
                        # Compute chamfer distance on vertices
                        pred_vertices_t = torch.tensor(pred_verts, dtype=torch.float32).unsqueeze(0).to(device)
                        gt_vertices_t = torch.tensor(gt_verts, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        pred_ptcld = Pointclouds(points=pred_vertices_t)
                        gt_ptcld = Pointclouds(points=gt_vertices_t)
                        
                        errors, _ = chamfer_distance(pred_ptcld, gt_ptcld, batch_reduction=None, point_reduction=None)
                        error_pred2gt = errors[0]
                        distance_pred2gt = torch.sqrt(error_pred2gt) * 100.0
                        
                        # Squeeze to remove batch dimension: (1, N) -> (N,)
                        distance_np = distance_pred2gt.squeeze(0).cpu().numpy()
                        all_distances.append(distance_np)
                        
                        up2you_data.append({
                            'gt_vertices': gt_verts,
                            'gt_faces': gt_faces,
                            'pred_vertices': pred_verts,
                            'pred_faces': pred_faces,
                            'distance': distance_np,
                            'label': label_base
                        })
                    else:
                        up2you_data.append(None)
                except Exception as e:
                    print(f'Error processing UP2You {take_dir}: {e}')
                    up2you_data.append(None)
            else:
                up2you_data.append(None)
        
        # Load Ours mesh from scaled directory for error computation
        if 'Ours' in rows_to_plot:
            ours_filepath = os.path.join(scaled_path, 'ours_scaled.obj')
            gt_filepath = os.path.join(scaled_path, 'gt_scaled.obj')
            
            if os.path.exists(ours_filepath) and os.path.exists(gt_filepath):
                try:
                    pred_verts, pred_faces, _ = load_obj_colored(ours_filepath)
                    gt_verts, gt_faces, _ = load_obj_colored(gt_filepath)
                    
                    if pred_verts is not None and gt_verts is not None:
                        # Compute chamfer distance on vertices
                        pred_vertices_t = torch.tensor(pred_verts, dtype=torch.float32).unsqueeze(0).to(device)
                        gt_vertices_t = torch.tensor(gt_verts, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        pred_ptcld = Pointclouds(points=pred_vertices_t)
                        gt_ptcld = Pointclouds(points=gt_vertices_t)
                        
                        errors, _ = chamfer_distance(pred_ptcld, gt_ptcld, batch_reduction=None, point_reduction=None)
                        error_pred2gt = errors[0]
                        distance_pred2gt = torch.sqrt(error_pred2gt) * 100.0
                        
                        # Squeeze to remove batch dimension: (1, N) -> (N,)
                        distance_np = distance_pred2gt.squeeze(0).cpu().numpy()
                        all_distances.append(distance_np)
                        
                        ours_data.append({
                            'gt_vertices': gt_verts,
                            'gt_faces': gt_faces,
                            'pred_vertices': pred_verts,
                            'pred_faces': pred_faces,
                            'distance': distance_np,
                            'label': label_base
                        })
                    else:
                        ours_data.append(None)
                except Exception as e:
                    print(f'Error processing Ours {take_dir}: {e}')
                    ours_data.append(None)
            else:
                ours_data.append(None)
    
    # Compute global min/max
    if len(all_distances) > 0:
        all_distances_flat = np.concatenate([d.flatten() for d in all_distances])
        global_min = all_distances_flat.min()
        global_max = all_distances_flat.max()
    else:
        global_min, global_max = 0.0, 1.0
    
    print(f'Global distance range: [{global_min:.4f}, {global_max:.4f}] cm')
    
    return up2you_data, ours_data, gt_data, global_min, global_max

def load_normals_meshes(base_dir, stride=1, rows_to_plot=['GT', 'UP2You', 'Ours']):
    """Load meshes from main Take* directories for normals visualization."""
    gt_geoms = []
    ours_geoms = []
    up2you_geoms = []
    
    # Find all Take* subdirectories
    try:
        all_items = os.listdir(base_dir)
        take_dirs = [d for d in all_items if d.startswith('Take') and os.path.isdir(os.path.join(base_dir, d))]
        
        def take_sort_key(name):
            try:
                parts = name.replace('Take', '').split('_')
                if parts[0].isdigit():
                    return (0, int(parts[0]))  # Numeric: sort by number
                else:
                    return (1, name)  # Non-numeric: sort alphabetically after numeric
            except (ValueError, IndexError):
                return (1, name)
        take_dirs = sorted(take_dirs, key=take_sort_key)
    except Exception as e:
        print(f"Error listing directory {base_dir}: {e}")
        take_dirs = []
    
    # Process each Take* directory
    for take_dir in take_dirs[::stride]:
        take_path = os.path.join(base_dir, take_dir)
        label_base = take_dir
        
        # Load GT mesh (gt.ply or gt.obj)
        if 'GT' in rows_to_plot:
            gt_filepath_ply = os.path.join(take_path, 'gt.ply')
            gt_filepath_obj = os.path.join(take_path, 'gt.obj')
            gt_filepath = None
            
            if os.path.exists(gt_filepath_ply):
                gt_filepath = gt_filepath_ply
            elif os.path.exists(gt_filepath_obj):
                gt_filepath = gt_filepath_obj
            
            if gt_filepath is not None:
                try:
                    if gt_filepath.endswith('.ply'):
                        verts, faces, colors = load_ply_colored(gt_filepath)
                    else:
                        verts, faces, colors = load_obj_colored(gt_filepath)
                    
                    if verts is not None:
                        # Compute surface normals and convert to colors
                        if faces is not None and len(faces) > 0:
                            vertex_normals = compute_vertex_normals(verts, faces)
                            if vertex_normals is not None:
                                colors = normals_to_colors(vertex_normals)
                        gt_geoms.append((verts, faces, colors, label_base))
                    else:
                        gt_geoms.append(None)
                except Exception as e:
                    print(f'Error loading GT {take_dir}: {e}')
                    gt_geoms.append(None)
            else:
                gt_geoms.append(None)
        
        # Load Ours mesh (ours.obj)
        if 'Ours' in rows_to_plot:
            ours_filepath = os.path.join(take_path, 'ours.obj')
            if os.path.exists(ours_filepath):
                try:
                    verts, faces, colors = load_obj_colored(ours_filepath)
                    if verts is not None:
                        # Compute surface normals and convert to colors
                        if faces is not None and len(faces) > 0:
                            vertex_normals = compute_vertex_normals(verts, faces)
                            if vertex_normals is not None:
                                colors = normals_to_colors(vertex_normals)
                        ours_geoms.append((verts, faces, colors, label_base))
                    else:
                        ours_geoms.append(None)
                except Exception as e:
                    print(f'Error loading Ours {take_dir}: {e}')
                    ours_geoms.append(None)
            else:
                ours_geoms.append(None)
        
        # Load UP2You mesh (up2you.obj)
        if 'UP2You' in rows_to_plot:
            up2you_filepath = os.path.join(take_path, 'up2you.obj')
            if os.path.exists(up2you_filepath):
                try:
                    verts, faces, colors = load_obj_colored(up2you_filepath)
                    if verts is not None:
                        # Compute surface normals and convert to colors
                        if faces is not None and len(faces) > 0:
                            vertex_normals = compute_vertex_normals(verts, faces)
                            if vertex_normals is not None:
                                colors = normals_to_colors(vertex_normals)
                        up2you_geoms.append((verts, faces, colors, label_base))
                    else:
                        up2you_geoms.append(None)
                except Exception as e:
                    print(f'Error loading UP2You {take_dir}: {e}')
                    up2you_geoms.append(None)
            else:
                up2you_geoms.append(None)
    
    return gt_geoms, ours_geoms, up2you_geoms

def render_with_error_colors(data, global_min, global_max, colormap, canvas_size=(1200, 800), center=None, camera_pos=None, max_range=None):
    """Render mesh with global color scale for error visualization."""
    if data is None:
        return None
    
    distance = data['distance']
    pred_vertices = data['pred_vertices']
    pred_faces = data['pred_faces']
    
    # Flatten distance array to match number of points
    if distance.ndim > 1:
        distance = distance.flatten()
    
    # Ensure distance matches number of vertices
    if len(distance) != len(pred_vertices):
        if len(distance) < len(pred_vertices):
            distance = np.pad(distance, (0, len(pred_vertices) - len(distance)), mode='edge')
        else:
            distance = distance[:len(pred_vertices)]
    
    # Normalize using global scale
    distance_normalized = (distance - global_min) / (global_max - global_min + 1e-8)
    distance_normalized = np.clip(distance_normalized, 0, 1)
    
    # Apply colormap
    if distance_normalized.ndim == 0:
        colors = colormap(distance_normalized)[:3].reshape(1, 3)
    else:
        colors = colormap(distance_normalized)[:, :3]
    
    # Get view parameters if not provided
    if center is None or camera_pos is None or max_range is None:
        center, camera_pos, max_range = get_view_params(pred_vertices)
    
    # Render mesh
    rendered_img = render_mesh_pyvista(
        verts=pred_vertices,
        faces=pred_faces,
        colors=colors,
        center=center,
        camera_pos=camera_pos,
        max_range=max_range,
        opacity=1.0,
        default_color='gray',
        canvas_size=canvas_size
    )
    
    # Keep RGBA format
    if rendered_img.shape[2] == 3:
        rendered_img = np.concatenate([rendered_img, np.full((*rendered_img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
    
    return rendered_img

def create_interleaved_figure(gt_normals_images, up2you_normals_images, ours_normals_images,
                              gt_error_images, up2you_error_images, ours_error_images,
                              gt_rotated_images, up2you_rotated_images, ours_rotated_images,
                              gt_labels, up2you_labels, ours_labels,
                              global_min, global_max, rows_to_plot=['GT', 'UP2You', 'Ours'],
                              output_path='Figures/vs_comparison.png', save_dpi=300):
    """Create a figure with interleaving columns: normals, error maps, and rotated views."""
    # Determine number of subjects (each subject has 3 columns: normals + error + rotated)
    num_subjects = max(
        len(gt_normals_images) if gt_normals_images else 0,
        len(up2you_normals_images) if up2you_normals_images else 0,
        len(ours_normals_images) if ours_normals_images else 0
    )
    
    if num_subjects == 0:
        print("No images to plot!")
        return
    
    num_cols = num_subjects * 3  # Each subject has 3 columns: normals + error + rotated
    num_rows = len(rows_to_plot)
    subfig_size = 4
    
    # Create figure
    fig = plt.figure(figsize=(subfig_size * num_cols, subfig_size * num_rows * 1.4))
    fig.patch.set_alpha(0)  # Make figure background transparent
    
    # Map row names to their data
    row_data = {
        'GT': {
            'normals': gt_normals_images,
            'errors': gt_error_images,
            'rotated': gt_rotated_images,
            'labels': gt_labels,
            'label': ''
        },
        'UP2You': {
            'normals': up2you_normals_images,
            'errors': up2you_error_images,
            'rotated': up2you_rotated_images,
            'labels': up2you_labels,
            'label': ''
        },
        'Ours': {
            'normals': ours_normals_images,
            'errors': ours_error_images,
            'rotated': ours_rotated_images,
            'labels': ours_labels,
            'label': ''
        }
    }
    
    # Create axes for each row with interleaving columns
    row_axes_dict = {}
    for row_idx, row_name in enumerate(rows_to_plot):
        if row_name not in row_data:
            continue
        
        data = row_data[row_name]
        axes_list = []
        
        for subject_idx in range(num_subjects):
            # Column for normals
            col_idx_normals = subject_idx * 3
            ax_normals = fig.add_subplot(num_rows, num_cols, row_idx * num_cols + col_idx_normals + 1)
            
            if subject_idx < len(data['normals']) and data['normals'][subject_idx] is not None:
                ax_normals.imshow(data['normals'][subject_idx])
            else:
                ax_normals.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_normals.transAxes)
            
            ax_normals.axis('off')
            ax_normals.patch.set_alpha(0)
            axes_list.append(ax_normals)
            
            # Column for error map
            col_idx_error = subject_idx * 3 + 1
            ax_error = fig.add_subplot(num_rows, num_cols, row_idx * num_cols + col_idx_error + 1)
            
            if subject_idx < len(data['errors']) and data['errors'][subject_idx] is not None:
                ax_error.imshow(data['errors'][subject_idx])
            else:
                ax_error.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_error.transAxes)
            
            ax_error.axis('off')
            ax_error.patch.set_alpha(0)
            axes_list.append(ax_error)
            
            # Column for rotated view
            col_idx_rotated = subject_idx * 3 + 2
            ax_rotated = fig.add_subplot(num_rows, num_cols, row_idx * num_cols + col_idx_rotated + 1)
            
            if subject_idx < len(data['rotated']) and data['rotated'][subject_idx] is not None:
                ax_rotated.imshow(data['rotated'][subject_idx])
            else:
                ax_rotated.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_rotated.transAxes)
            
            ax_rotated.axis('off')
            ax_rotated.patch.set_alpha(0)
            axes_list.append(ax_rotated)
        
        if len(axes_list) > 0:
            row_axes_dict[row_name] = axes_list
    
    # Apply tight layout
    plt.tight_layout(pad=0.1)
    
    # Position columns: group each subject tightly, with whitespace between subjects
    try:
        for row_name in rows_to_plot:
            axes_list = row_axes_dict.get(row_name, [])
            if len(axes_list) > 0:
                # Get reference position from first normals column
                pos0 = axes_list[0].get_position()
                width = pos0.width
                height = pos0.height
                y0 = pos0.y0
                x0_start = pos0.x0
                
                # Shift all plots to the right to give more space for multi-line labels
                label_space_offset = width * 0.15  # Shift plots right by 15% of column width
                x0_start = x0_start + label_space_offset
                

                # Tight spacing between columns within a subject (small overlap or minimal gap)
                overlap_fraction = 0.3  # 25% overlap between columns within a subject
                spacing_within_subject = width * (1.0 - overlap_fraction)  # Step between columns within same subject
                
                # Reduced spacing between subject groups
                spacing_between_subjects = width * 0.1  # Small spacing between subjects
                # Total span of one subject group: 3 columns with overlaps + spacing to next subject
                # After overlaps, each subject spans approximately: width + 2*spacing_within_subject
                subject_group_width = width + 2 * spacing_within_subject
                step_between_subjects = subject_group_width + spacing_between_subjects
                
                # Process each subject's three columns
                for subject_idx in range(num_subjects):
                    normals_idx = subject_idx * 3
                    error_idx = subject_idx * 3 + 1
                    rotated_idx = subject_idx * 3 + 2
                    
                    if normals_idx < len(axes_list) and error_idx < len(axes_list) and rotated_idx < len(axes_list):
                        # Calculate x position for this subject group
                        x0_normals = x0_start + subject_idx * step_between_subjects
                        x0_error = x0_normals + spacing_within_subject
                        x0_rotated = x0_error + spacing_within_subject
                        
                        # Set positions
                        axes_list[normals_idx].set_position([x0_normals, y0, width, height])
                        axes_list[error_idx].set_position([x0_error, y0, width, height])
                        axes_list[rotated_idx].set_position([x0_rotated, y0, width, height])
    except Exception as e:
        print(f"Error in applying positioning: {e}")
        pass
    
    # Add row labels at the top-left corner of each row
    try:
        label_fontsize = 20
        label_offset_x = 0.005
        label_offset_y = 0.002
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
    
    # Add global colorbar at the right edge for error maps (spans only last two rows)
    try:
        colormap = plt.colormaps.get_cmap('RdYlGn_r')
    except AttributeError:
        colormap = plt.cm.get_cmap('RdYlGn_r')
    
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    
    # Position colorbar at the right edge (spans only last two rows: UP2You and Ours)
    # Find the last two rows (skip GT if present)
    second_to_last_row_axes = None
    last_row_axes = None
    row_names_list = list(rows_to_plot)
    
    # Get the last two rows in reverse order
    for row_name in reversed(row_names_list):
        axes_list = row_axes_dict.get(row_name, [])
        if len(axes_list) > 0:
            if last_row_axes is None:
                last_row_axes = axes_list
            elif second_to_last_row_axes is None:
                second_to_last_row_axes = axes_list
                break  # Found both rows, exit
    
    # If we only have one row with error maps, use just that row
    if last_row_axes and len(last_row_axes) > 0:
        pos_last = last_row_axes[-1].get_position()
        right_edge = pos_last.x0 + pos_last.width
        cbar_width = 0.008
        
        if second_to_last_row_axes and len(second_to_last_row_axes) > 0:
            # Span from second-to-last row to last row
            top_pos = second_to_last_row_axes[0].get_position()
            bottom_pos = last_row_axes[0].get_position()
            cbar_height = (top_pos.y0 + top_pos.height) - bottom_pos.y0
            cbar_y0 = bottom_pos.y0
        else:
            # Only one row, use just that row's height
            cbar_height = last_row_axes[0].get_position().height
            cbar_y0 = last_row_axes[0].get_position().y0
        
        cbar_ax = fig.add_axes([right_edge + 0.01, cbar_y0, cbar_width, cbar_height])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Chamfer distance (cm)', rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=14)  # Make tick labels slightly larger too
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=save_dpi, bbox_inches='tight', pad_inches=0.1, transparent=True)
    print(f'Saved comparison figure to: {output_path}')
    plt.close()

def main(base_dir, stride=1, rows_to_plot=['GT', 'UP2You', 'Ours'], output_path='Figures/vs_comparison.png', save_dpi=300, canvas_size=(1200, 800)):
    """Main function to create interleaved normals and error comparison figure.
    
    Args:
        base_dir: Base directory containing Take* subdirectories
        stride: Stride for selecting samples
        rows_to_plot: List of rows to plot, e.g., ['GT', 'UP2You', 'Ours']
        output_path: Path to save the output figure
        save_dpi: DPI for saving the figure
        canvas_size: Canvas size for rendering
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Plotting rows: {rows_to_plot}')
    
    # Load meshes for normals visualization (from main Take* directories)
    print('Loading meshes for normals visualization...')
    gt_normals_geoms, ours_normals_geoms, up2you_normals_geoms = load_normals_meshes(
        base_dir, stride=stride, rows_to_plot=rows_to_plot
    )
    
    # Compute distances and load meshes for error visualization (from scaled directories)
    print('Computing distances and loading meshes for error visualization...')
    up2you_error_data, ours_error_data, gt_error_data, global_min, global_max = compute_all_distances(
        base_dir, stride=stride, device=device, rows_to_plot=rows_to_plot
    )
    
    print(f'Rendering with global error scale [{global_min:.4f}, {global_max:.4f}] cm...')
    # Use new matplotlib API to avoid deprecation warning
    try:
        colormap = plt.colormaps.get_cmap('RdYlGn_r')
    except AttributeError:
        colormap = plt.cm.get_cmap('RdYlGn_r')
    
    # Render normals images
    gt_normals_images = []
    up2you_normals_images = []
    ours_normals_images = []
    gt_labels = []
    up2you_labels = []
    ours_labels = []
    
    if 'GT' in rows_to_plot:
        # GT row: normals, gray_front, gray_back
        for idx, geom in enumerate(gt_normals_geoms):
            if geom is not None:
                verts, faces, colors, label = geom
                center, camera_pos, max_range = get_view_params(verts)
                # Render normals (first column)
                img = render_mesh_pyvista(verts, faces, colors=colors, center=center, camera_pos=camera_pos,
                                         max_range=max_range, opacity=1.0, default_color='lightgray', canvas_size=canvas_size)
                if img.shape[2] == 3:
                    img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
                gt_normals_images.append(img)
                gt_labels.append(label)
            else:
                gt_normals_images.append(None)
                if idx < len(gt_error_data) and gt_error_data[idx] is not None:
                    gt_labels.append(gt_error_data[idx]['label'])
                else:
                    gt_labels.append('')
    
    if 'UP2You' in rows_to_plot:
        # UP2You row: normals, error_front, error_back
        for idx, geom in enumerate(up2you_normals_geoms):
            if geom is not None:
                verts, faces, colors, label = geom
                center, camera_pos, max_range = get_view_params(verts)
                # Render normals (first column)
                img = render_mesh_pyvista(verts, faces, colors=colors, center=center, camera_pos=camera_pos,
                                         max_range=max_range, opacity=1.0, default_color='lightgray', canvas_size=canvas_size)
                if img.shape[2] == 3:
                    img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
                up2you_normals_images.append(img)
                up2you_labels.append(label)
            else:
                up2you_normals_images.append(None)
                if idx < len(up2you_error_data) and up2you_error_data[idx] is not None:
                    up2you_labels.append(up2you_error_data[idx]['label'])
                else:
                    up2you_labels.append('')
    
    if 'Ours' in rows_to_plot:
        # Ours row: normals, error_front, error_back
        for idx, geom in enumerate(ours_normals_geoms):
            if geom is not None:
                verts, faces, colors, label = geom
                center, camera_pos, max_range = get_view_params(verts)
                # Render normals (first column)
                img = render_mesh_pyvista(verts, faces, colors=colors, center=center, camera_pos=camera_pos,
                                         max_range=max_range, opacity=1.0, default_color='lightgray', canvas_size=canvas_size)
                if img.shape[2] == 3:
                    img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
                ours_normals_images.append(img)
                ours_labels.append(label)
            else:
                ours_normals_images.append(None)
                if idx < len(ours_error_data) and ours_error_data[idx] is not None:
                    ours_labels.append(ours_error_data[idx]['label'])
                else:
                    ours_labels.append('')
    
    # Render error images
    gt_error_images = []
    up2you_error_images = []
    ours_error_images = []
    
    if 'GT' in rows_to_plot:
        for data in gt_error_data:
            if data is not None:
                center, camera_pos, max_range = get_view_params(data['gt_vertices'])
                img = render_mesh_pyvista(data['gt_vertices'], data['gt_faces'], colors=None,
                                         center=center, camera_pos=camera_pos, max_range=max_range,
                                         opacity=1.0, default_color='lightgray', canvas_size=canvas_size)
                if img.shape[2] == 3:
                    img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
                gt_error_images.append(img)
            else:
                gt_error_images.append(None)
    
    if 'UP2You' in rows_to_plot:
        for data in up2you_error_data:
            if data is not None:
                img = render_with_error_colors(data, global_min, global_max, colormap, canvas_size=canvas_size)
                up2you_error_images.append(img)
            else:
                up2you_error_images.append(None)
    
    if 'Ours' in rows_to_plot:
        for data in ours_error_data:
            if data is not None:
                img = render_with_error_colors(data, global_min, global_max, colormap, canvas_size=canvas_size)
                ours_error_images.append(img)
            else:
                ours_error_images.append(None)
    
    # Render rotated images (180 degrees)
    gt_rotated_images = []
    up2you_rotated_images = []
    ours_rotated_images = []
    
    if 'GT' in rows_to_plot:
        for data in gt_error_data:
            if data is not None:
                # GT row: show gray mesh in rotated view
                center, camera_pos, max_range = get_view_params_rotated(data['gt_vertices'], rotation_degrees=180)
                img = render_mesh_pyvista(data['gt_vertices'], data['gt_faces'], colors=None,
                                         center=center, camera_pos=camera_pos, max_range=max_range,
                                         opacity=1.0, default_color='lightgray', canvas_size=canvas_size)
                if img.shape[2] == 3:
                    img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
                gt_rotated_images.append(img)
            else:
                gt_rotated_images.append(None)
    
    if 'UP2You' in rows_to_plot:
        for data in up2you_error_data:
            if data is not None:
                # UP2You row: show error map in rotated view
                center, camera_pos, max_range = get_view_params_rotated(data['pred_vertices'], rotation_degrees=180)
                img = render_with_error_colors(data, global_min, global_max, colormap, 
                                              canvas_size=canvas_size, center=center, camera_pos=camera_pos, max_range=max_range)
                up2you_rotated_images.append(img)
            else:
                up2you_rotated_images.append(None)
    
    if 'Ours' in rows_to_plot:
        for data in ours_error_data:
            if data is not None:
                # Ours row: show error map in rotated view
                center, camera_pos, max_range = get_view_params_rotated(data['pred_vertices'], rotation_degrees=180)
                img = render_with_error_colors(data, global_min, global_max, colormap,
                                              canvas_size=canvas_size, center=center, camera_pos=camera_pos, max_range=max_range)
                ours_rotated_images.append(img)
            else:
                ours_rotated_images.append(None)
    
    print(f'Rendered {sum(1 for img in gt_normals_images if img is not None)} GT normals images')
    print(f'Rendered {sum(1 for img in up2you_normals_images if img is not None)} UP2You normals images')
    print(f'Rendered {sum(1 for img in ours_normals_images if img is not None)} Ours normals images')
    print(f'Rendered {sum(1 for img in gt_error_images if img is not None)} GT error images')
    print(f'Rendered {sum(1 for img in up2you_error_images if img is not None)} UP2You error images')
    print(f'Rendered {sum(1 for img in ours_error_images if img is not None)} Ours error images')
    print(f'Rendered {sum(1 for img in gt_rotated_images if img is not None)} GT rotated images')
    print(f'Rendered {sum(1 for img in up2you_rotated_images if img is not None)} UP2You rotated images')
    print(f'Rendered {sum(1 for img in ours_rotated_images if img is not None)} Ours rotated images')
    
    # Compute and print chamfer distances for each subject
    print('\n' + '='*60)
    print('Chamfer distances per subject:')
    print('='*60)
    
    # Get labels for subjects (use GT labels as reference)
    subject_labels = []
    if 'GT' in rows_to_plot and len(gt_labels) > 0:
        subject_labels = gt_labels
    elif 'UP2You' in rows_to_plot and len(up2you_labels) > 0:
        subject_labels = up2you_labels
    elif 'Ours' in rows_to_plot and len(ours_labels) > 0:
        subject_labels = ours_labels
    
    max_subjects = max(
        len(up2you_error_data) if 'UP2You' in rows_to_plot else 0,
        len(ours_error_data) if 'Ours' in rows_to_plot else 0
    )
    
    up2you_means = []
    ours_means = []
    
    for idx in range(max_subjects):
        label = subject_labels[idx] if idx < len(subject_labels) else f'Subject {idx+1}'
        
        line_parts = [f'{label:30s}']
        
        # UP2You distance
        if 'UP2You' in rows_to_plot:
            if idx < len(up2you_error_data) and up2you_error_data[idx] is not None:
                mean_dist = up2you_error_data[idx]['distance'].mean()
                up2you_means.append(mean_dist)
                line_parts.append(f'UP2You: {mean_dist:.4f} cm')
            else:
                line_parts.append('UP2You: N/A')
        
        # Ours distance
        if 'Ours' in rows_to_plot:
            if idx < len(ours_error_data) and ours_error_data[idx] is not None:
                mean_dist = ours_error_data[idx]['distance'].mean()
                ours_means.append(mean_dist)
                line_parts.append(f'Ours: {mean_dist:.4f} cm')
            else:
                line_parts.append('Ours: N/A')
        
        print(' | '.join(line_parts))
    
    # Print overall means
    print('='*60)
    if len(up2you_means) > 0:
        up2you_overall_mean = np.mean(up2you_means)
        print(f'UP2You overall mean: {up2you_overall_mean:.4f} cm (across {len(up2you_means)} samples)')
    else:
        print('UP2You: No valid data')
    
    if len(ours_means) > 0:
        ours_overall_mean = np.mean(ours_means)
        print(f'Ours overall mean: {ours_overall_mean:.4f} cm (across {len(ours_means)} samples)')
    else:
        print('Ours: No valid data')
    print('='*60)
    
    print('Creating interleaved comparison figure...')
    create_interleaved_figure(
        gt_normals_images, up2you_normals_images, ours_normals_images,
        gt_error_images, up2you_error_images, ours_error_images,
        gt_rotated_images, up2you_rotated_images, ours_rotated_images,
        gt_labels, up2you_labels, ours_labels,
        global_min, global_max, rows_to_plot=rows_to_plot, output_path=output_path, save_dpi=save_dpi
    )
    print('Done!')

    ''' ---------------------------------------------------------- '''
if __name__ == '__main__':
    save_dpi = 200
    canvas_size = (900, 600)

    # Configuration: Select which rows to plot
    # Options: 'GT', 'UP2You', 'Ours'
    rows_to_plot = ['GT', 'UP2You', 'Ours']
    
    base_dir = 'Figures/vs_up2you'
    output_path = os.path.join(base_dir, 'vs_comparison_fb.png')

    # Number of subplots per row (stride=1 means all, stride=2 means every 2nd, etc.)
    stride = 1

    main(base_dir=base_dir, stride=stride, rows_to_plot=rows_to_plot, output_path=output_path, save_dpi=save_dpi, canvas_size=canvas_size)

import os
import numpy as np
import trimesh

try:
    import pyvista as pv
    PV_AVAILABLE = True
except ImportError:
    PV_AVAILABLE = False

def load_ply_pointcloud(filepath):
    """Load PLY file as point cloud (vertices only)."""
    try:
        # First try loading without force='mesh' to handle point clouds
        loaded = trimesh.load(filepath)
        
        # Check if it's a PointCloud object
        if isinstance(loaded, trimesh.PointCloud):
            return np.asarray(loaded.vertices)
        
        # Check if it's a Trimesh with vertices
        if hasattr(loaded, 'vertices'):
            verts = np.asarray(loaded.vertices)
            # If it has no faces or very few faces, treat as point cloud
            if not hasattr(loaded, 'faces') or loaded.faces is None or len(loaded.faces) == 0:
                return verts
            # If it has faces but we want vertices, return vertices anyway
            return verts
        
        # Try with force='mesh' as fallback
        mesh = trimesh.load(filepath, force='mesh')
        if hasattr(mesh, 'vertices'):
            verts = np.asarray(mesh.vertices)
            if len(verts) > 0:
                return verts
    except Exception as e:
        print(f"Warning: trimesh failed to load {filepath}: {e}")
    
    # Fallback to PyVista if available
    if PV_AVAILABLE:
        try:
            mesh = pv.read(filepath)
            points = np.asarray(mesh.points)
            if len(points) > 0:
                return points
        except Exception as e:
            print(f"Warning: PyVista failed to load {filepath}: {e}")
    
    print(f"Error: Could not load point cloud from {filepath}")
    return None

def load_mesh(filepath):
    """Load mesh file (PLY, OBJ, etc.) with faces."""
    try:
        mesh = trimesh.load(filepath, force='mesh')
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            return mesh
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def compute_point_to_surface_distance(points, mesh):
    """
    Compute point-to-surface distance from points to mesh.
    
    Args:
        points: Nx3 numpy array of points
        mesh: trimesh.Trimesh object
    
    Returns:
        distances: N-length numpy array of distances (in same units as mesh)
        mean_distance: mean distance
        median_distance: median distance
    """
    if len(points) == 0:
        return np.array([]), 0.0, 0.0
    
    # Find nearest points on mesh surface
    closest_points, distances, face_indices = mesh.nearest.on_surface(points)
    
    # distances is already the point-to-surface distance
    distances = np.asarray(distances)
    
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    
    return distances, mean_dist, median_dist

def process_directory(path):
    """
    Scan directory for GT meshes and predicted point clouds, compute point-to-surface distances.
    
    Args:
        path: Directory path to scan
    """
    if not os.path.exists(path):
        print(f"Error: Path {path} does not exist")
        return
    
    # Find GT mesh files (files starting with 'gt_vp_' or similar patterns)
    gt_files = []
    pred_files = []
    
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if not os.path.isfile(filepath):
            continue
        
        # Look for GT meshes (could be PLY with faces, OBJ, etc.)
        if filename.startswith('gt_vp_') and (filename.endswith('.ply') or filename.endswith('.obj')):
            gt_files.append((filename, filepath))
        
        # Look for predicted point clouds
        if filename.startswith('pred_vp_init_') and filename.endswith('.ply'): # and not filename.startswith('pred_vp_init_'):
            pred_files.append((filename, filepath))
    
    # Sort files by timestep
    def sort_key(name):
        base = os.path.splitext(name)[0]
        parts = base.split('_')
        if len(parts) > 0:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return base
    
    gt_files = sorted(gt_files, key=lambda x: sort_key(x[0]))
    pred_files = sorted(pred_files, key=lambda x: sort_key(x[0]))
    
    # Limit to 50 items if more than 50 are found
    original_num_pred = len(pred_files)
    if len(pred_files) > 50:
        pred_files = pred_files[:50]
        print(f"Found {original_num_pred} predicted point cloud files, limiting to first 50")
    else:
        print(f"Found {len(pred_files)} predicted point cloud files")
    
    print(f"Found {len(gt_files)} GT mesh files")
    print(f"Processing {len(pred_files)} predicted point cloud files")
    
    # Match GT and predicted files by timestep
    results = []
    
    for pred_filename, pred_path in pred_files:
        # Extract timestep from predicted filename (e.g., 'pred_vp_000.ply' -> '000')
        pred_base = os.path.splitext(pred_filename)[0]
        pred_parts = pred_base.split('_')
        if len(pred_parts) < 3:
            continue
        timestep = pred_parts[-1]
        
        # Find matching GT file
        gt_filename = None
        gt_path = None
        for gt_fname, gt_fpath in gt_files:
            if f'_{timestep}.' in gt_fname:
                gt_filename = gt_fname
                gt_path = gt_fpath
                break
        
        if gt_path is None:
            print(f"Warning: No matching GT mesh found for {pred_filename}")
            continue
        
        # Load GT mesh
        gt_mesh = load_mesh(gt_path)
        if gt_mesh is None:
            print(f"Warning: Could not load GT mesh {gt_filename}")
            continue
        
        # Load predicted point cloud
        print(f"Loading predicted point cloud: {pred_filename}")
        pred_points = load_ply_pointcloud(pred_path)
        if pred_points is None:
            print(f"Warning: Could not load predicted point cloud {pred_filename}")
            continue
        
        if len(pred_points) == 0:
            print(f"Warning: Loaded point cloud {pred_filename} has 0 points")
            continue
        
        print(f"Processing: {pred_filename} -> {gt_filename}")
        print(f"  GT mesh: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.faces)} faces")
        print(f"  Predicted points: {len(pred_points)} points")
        
        # Compute point-to-surface distances
        distances, mean_dist, median_dist = compute_point_to_surface_distance(pred_points, gt_mesh)
        
        # Convert to centimeters (assuming mesh is in meters)
        mean_dist_cm = mean_dist * 100.0
        median_dist_cm = median_dist * 100.0
        std_dist_cm = np.std(distances) * 100.0
        
        results.append({
            'pred_file': pred_filename,
            'gt_file': gt_filename,
            'timestep': timestep,
            'num_points': len(pred_points),
            'mean_distance_cm': mean_dist_cm,
            'median_distance_cm': median_dist_cm,
            'std_distance_cm': std_dist_cm,
            'min_distance_cm': np.min(distances) * 100.0,
            'max_distance_cm': np.max(distances) * 100.0
        })
        
        print(f"  Mean distance: {mean_dist_cm:.3f} cm")
        print(f"  Median distance: {median_dist_cm:.3f} cm")
        print(f"  Std distance: {std_dist_cm:.3f} cm")
        print()
    
    # Print summary
    if len(results) > 0:
        print("=" * 60)
        print("Summary:")
        print("=" * 60)
        all_means = [r['mean_distance_cm'] for r in results]
        all_medians = [r['median_distance_cm'] for r in results]
        
        print(f"Overall mean distance: {np.mean(all_means):.3f} cm")
        print(f"Overall median distance: {np.mean(all_medians):.3f} cm")
        print(f"Std of mean distances: {np.std(all_means):.3f} cm")
        print()
        
        print("Per-file results:")
        for r in results:
            print(f"  {r['pred_file']}: mean={r['mean_distance_cm']:.3f} cm, "
                  f"median={r['median_distance_cm']:.3f} cm")
    
    return results

if __name__ == "__main__":
    path = 'exp/exp_sync/vis/for_visuals'
    results = process_directory(path)

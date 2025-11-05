path = 'vis/00134_take1/pred_vp_050.ply'

import open3d as o3d
import trimesh
import numpy as np
import os
from scipy.spatial import cKDTree

def keep_largest(mesh):
    mesh_lst = mesh.split(only_watertight=False)
    if len(mesh_lst) == 0:
        return mesh
    keep_mesh = mesh_lst[0]
    for mesh in mesh_lst:
        if mesh.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = mesh
    return keep_mesh


def smooth_point_cloud(points, k=8, iterations=3):
    """
    Smooth point cloud using Laplacian smoothing
    Args:
        points: Nx3 numpy array of points
        k: number of neighbors to consider
        iterations: number of smoothing iterations
    """
    tree = cKDTree(points)
    smoothed_points = points.copy()
    
    for _ in range(iterations):
        # Find k-nearest neighbors
        _, indices = tree.query(smoothed_points, k=k+1)
        
        # Compute new positions as average of neighbors
        new_points = np.zeros_like(smoothed_points)
        for i in range(len(smoothed_points)):
            neighbors = smoothed_points[indices[i, 1:]]  # exclude self
            new_points[i] = np.mean(neighbors, axis=0)
        
        # Update points
        smoothed_points = new_points
    
    return smoothed_points
def poisson(path, depth=9, decimation=False):
    # Load the point cloud from a file first
    pcl = o3d.io.read_point_cloud(path)
    
    if len(pcl.points) == 0:
        raise ValueError(f"Point cloud {path} is empty!")
    
    print(f"Loaded point cloud with {len(pcl.points)} points")
    
    # Pre-process point cloud
    # Remove duplicate points
    pcl = pcl.remove_duplicated_points()
    print(f"After removing duplicates: {len(pcl.points)} points")
    
    # Remove statistical outliers
    pcl, _ = pcl.remove_statistical_outlier(nb_neighbors=20, std_ratio=4.0)
    print(f"After removing outliers: {len(pcl.points)} points")
    
    if len(pcl.points) < 100:
        raise ValueError(f"Too few points remaining after preprocessing: {len(pcl.points)}")
    
    # Smooth point cloud before reconstruction
    print("Smoothing point cloud...")
    points = np.asarray(pcl.points)
    smoothed_points = smooth_point_cloud(points, k=8, iterations=3)
    pcl.points = o3d.utility.Vector3dVector(smoothed_points)
    print(f"Point cloud smoothed")
    
    # Estimate normals for the point cloud if not present
    if not pcl.has_normals():
        print("Estimating normals...")
        # Use adaptive radius based on point cloud size
        bbox = pcl.get_axis_aligned_bounding_box()
        diag = bbox.get_extent()
        radius = np.linalg.norm(diag) * 0.01  # 1% of diagonal
        radius = max(radius, 0.01)  # Minimum radius
        pcl.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        # Orient normals consistently
        pcl.orient_normals_consistent_tangent_plane(k=15)
    
    print(f"Running Poisson reconstruction with depth={depth}...")
    try:
        # Use fewer threads to avoid OpenBLAS issues
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcl, depth=depth, n_threads=1, width=0, scale=1.1, linear_fit=False
            )
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
        print("Trying with lower depth...")
        # Retry with lower depth
        depth = max(7, depth - 2)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcl, depth=depth, n_threads=1, width=0, scale=1.1, linear_fit=False
            )
    
    print(f"Reconstructed mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Remove low density vertices (likely outliers)
    if len(densities) > 0:
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"After removing low density vertices: {len(mesh.vertices)} vertices")
    
    # only keep the largest component
    mesh_trimesh = trimesh.Trimesh(np.array(mesh.vertices), np.array(mesh.triangles))
    largest_mesh = keep_largest(mesh_trimesh)
    
    if decimation:
        # mesh decimation for faster rendering
        print("Decimating mesh...")
        low_res_mesh = largest_mesh.simplify_quadric_decimation(50000)
        print(f"Decimated to {len(low_res_mesh.vertices)} vertices")
        return low_res_mesh
    else:
        return largest_mesh
    

if __name__ == "__main__":
    mesh_poisson = poisson(path)
    output_path = 'vis/00134_take1/pred_vp_050_poisson.obj'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh_poisson.export(output_path)
    print(f"Saved mesh to {output_path}")
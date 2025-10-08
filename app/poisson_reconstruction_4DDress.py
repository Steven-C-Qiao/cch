import os
import sys
import torch
import pickle

import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
from einops import rearrange

viridis = plt.colormaps.get_cmap('viridis')

from scipy.spatial import cKDTree


def remove_outliers(points, k=8, threshold=2.0):
    """
    Remove outliers using k-nearest neighbors statistics
    Args:
        points: Nx3 numpy array of points
        k: number of neighbors to consider
        threshold: standard deviation threshold for outlier removal
    """
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)  # k+1 because point is its own neighbor
    mean_dist = np.mean(distances[:, 1:], axis=1)  # exclude self-distance
    std_dist = np.std(distances[:, 1:], axis=1)
    
    # Remove points that are too far from their neighbors
    mask = mean_dist < (np.mean(mean_dist) + threshold * np.std(mean_dist))

    # import ipdb; ipdb.set_trace()
    return points[mask]

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

def denoise_point_cloud(points, k=8, threshold=2.0, iterations=3):
    """
    Apply both outlier removal and smoothing
    """
    # First remove outliers
    cleaned_points = remove_outliers(points, k=k, threshold=threshold)
    # Then smooth the remaining points
    smoothed_points = smooth_point_cloud(cleaned_points, k=k, iterations=iterations)
    return smoothed_points

def estimate_normals(pcd, k=30):
    """
    Estimate normals for the point cloud
    Args:
        pcd: Open3D point cloud
        k: number of neighbors to consider for normal estimation
    """
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k)
    )
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=100)
    
    return pcd

def poisson_reconstruction(pcd, depth=8, width=0, scale=1.1, linear_fit=False):
    """
    Perform Poisson surface reconstruction
    Args:
        pcd: Open3D point cloud with normals
        depth: maximum depth of the octree
        width: minimum width of the octree
        scale: scale factor for the reconstruction
        linear_fit: whether to use linear interpolation
    """
    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit
    )
    
    # Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh

if __name__=="__main__":
    with torch.no_grad():

        vp_list_path = 'app/avatar_outputs/vp_list.pkl'
        PREDS_PATH = 'app/avatar_outputs/preds.pkl'
        BATCH_PATH = 'app/avatar_outputs/init_batch.pkl'

        conf_threshold = 0.05

        with open(vp_list_path, 'rb') as f:
            vp_list = pickle.load(f)
        
        with open(PREDS_PATH, 'rb') as f:
            preds = pickle.load(f)
        
        with open(BATCH_PATH, 'rb') as f:
            batch = pickle.load(f)

        color = rearrange(batch['imgs'][0, :-1], 'n c h w -> n h w c')

        from tqdm import tqdm

        for i in tqdm(range(len(vp_list))):

            
            vp_bknhwc = vp_list[i]['vp'] # B, K, N, H, W, 3
            vc = vp_list[i]['vc']

            vp = vp_bknhwc[0, -1, ...] # N, H, W, 3

            mask_bkhw = batch['masks'] # B, K, H, W
            mask = mask_bkhw[0, :-1, ...].astype(bool) # N, H, W

            color_masked = color[mask].astype(np.float32)
            # color = (color * 255.).astype(np.uint8)


            vp_masked = vp[mask] # X, 3
            vp_masked = vp_masked.cpu().detach().numpy()

            # print(vp_masked.shape) # eg 103845, 3

            # Denoise point cloud
            vp_masked = denoise_point_cloud(vp_masked, k=8, threshold=10.0, iterations=1)

            # Convert to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vp_masked)
            pcd.colors = o3d.utility.Vector3dVector(color_masked)
    
            # Estimate normals
            pcd = estimate_normals(pcd, k=30)

            # Save point cloud with normals
            o3d.io.write_point_cloud(f"app/avatar_outputs/point_cloud_with_normals_{i}.ply", pcd)

            # Perform Poisson reconstruction
            mesh = poisson_reconstruction(
                pcd,
                depth=8,  # Adjust based on point cloud density
                scale=1.1,
                linear_fit=True
            )

            # Transfer normals from point cloud to mesh vertices
            # Create a KD tree for the point cloud points
            pcd_points = np.asarray(pcd.points)
            pcd_colors = np.asarray(pcd.colors)
            # Normalize the normals to use as colors
            # pcd_colors = (pcd_normals + 1) / 2  # Convert from [-1,1] to [0,1] range
            tree = cKDTree(pcd_points)

            # For each mesh vertex, find the nearest point cloud point and use its color
            mesh_vertices = np.asarray(mesh.vertices)
            distances, indices = tree.query(mesh_vertices)
            vertex_colors = pcd_colors[indices]

            # Assign colors to mesh
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            # Save the reconstructed mesh
            o3d.io.write_triangle_mesh(f"app/avatar_outputs/reconstructed_mesh_{i}.ply", mesh)

            # Optional: Visualize the result
            # o3d.visualization.draw_geometries([mesh])
            
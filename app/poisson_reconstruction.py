import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse 
import pickle

from einops import rearrange

from loguru import logger

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import look_at_view_transform

import matplotlib.pyplot as plt
viridis = plt.colormaps.get_cmap('viridis')

import sys 
sys.path.append('.')

from core.models.cch import CCH
from core.models.smpl import SMPL
from core.configs import paths
from core.configs.cch_cfg import get_cch_cfg_defaults
from core.utils.general_lbs import general_lbs
from core.utils.diffpointrend import PointCloudRenderer

MODEL_PATH = "/scratches/kyuban/cq244/CCH/cch/exp/exp_031_pred_w/saved_models/val_posed_loss_epoch=048.ckpt"
AMASS_POSE_PATH = '/scratches/kyuban/cq244/datasets/AMASS/CNRS/283/-01_L_1_stageii.npz'
CAPE_POSE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00134/longlong_ballet1_trial1'
AVATAR_PATH = '/scratches/kyuban/cq244/CCH/cch/avatar.pkl'


import open3d as o3d


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
        conf_threshold = 0.05

        with open(AVATAR_PATH, 'rb') as f:
            avatar = pickle.load(f)

        vc_pred = avatar['vc_pred'][0]
        vc_conf = avatar['vc_conf'][0]
        w_pred = avatar['w_pred'][0]
        joints = avatar['joints'][0]
        masks = avatar['masks'][0]
        conf_mask = (1/vc_conf) < conf_threshold

        full_mask = masks * conf_mask 

        vc_pred = rearrange(vc_pred, 'N H W C -> (N H W) C')
        vc_conf = rearrange(vc_conf, 'N H W -> (N H W)')
        w_pred = rearrange(w_pred, 'N H W J -> (N H W) J')
        full_mask = rearrange(full_mask, 'N H W -> (N H W)')

        vc_pred_masked = vc_pred[full_mask.astype(bool)]
        w_pred_masked = w_pred[full_mask.astype(bool)]

        # Denoise point cloud
        vc_pred_masked = denoise_point_cloud(vc_pred_masked, k=8, threshold=10.0, iterations=1)

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vc_pred_masked)

        # Estimate normals
        pcd = estimate_normals(pcd, k=30)

        # Save point cloud with normals
        o3d.io.write_point_cloud("tinkering/point_cloud_with_normals.ply", pcd)

        # Perform Poisson reconstruction
        mesh = poisson_reconstruction(
            pcd,
            depth=8,  # Adjust based on point cloud density
            scale=1.1,
            linear_fit=True
        )

        # Save the reconstructed mesh
        o3d.io.write_triangle_mesh("tinkering/reconstructed_mesh.ply", mesh)

        # Optional: Visualize the result
        o3d.visualization.draw_geometries([mesh])

        
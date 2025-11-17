import torch
import pytorch_lightning as pl
from einops import rearrange

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import sample_points_from_meshes

from core.utils.loss_utils import filter_by_quantile
from core.utils.general import check_and_fix_inf_nan

import open3d as o3d
import numpy as np
import trimesh

def keep_largest(mesh):
    mesh_lst = mesh.split(only_watertight=False)
    if len(mesh_lst) == 0:
        return mesh
    keep_mesh = mesh_lst[0]
    for mesh in mesh_lst:
        if mesh.vertices.shape[0] > keep_mesh.vertices.shape[0]:
            keep_mesh = mesh
    return keep_mesh


def poisson(points, depth=10, decimation=False):
    # Accept points as numpy array or o3d point cloud
    if isinstance(points, np.ndarray):
        # Validate and clean the numpy array
        if points.size == 0:
            raise ValueError("Point cloud is empty!")
        
        # Ensure correct shape (N, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must be shape (N, 3), got {points.shape}")
        
        # Make contiguous and ensure float64 dtype
        points = np.ascontiguousarray(points, dtype=np.float64)
        
        # Remove NaN and Inf values
        valid_mask = np.isfinite(points).all(axis=1)
        if not valid_mask.all():
            points = points[valid_mask]
            if len(points) == 0:
                raise ValueError("All points contain NaN or Inf values!")
        
        # Create Open3D point cloud
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
    elif isinstance(points, o3d.geometry.PointCloud):
        pcl = points
    elif isinstance(points, str):
        # Backward compatibility: if path is provided, load from file
        pcl = o3d.io.read_point_cloud(points)
    else:
        raise ValueError(f"Unsupported input type: {type(points)}")
    
    if len(pcl.points) == 0:
        raise ValueError(f"Point cloud is empty!")
    
    print(f"Loaded point cloud with {len(pcl.points)} points")
    pcl, _ = pcl.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
    print(f"After removing outliers: {len(pcl.points)} points")
    
    # Estimate normals for the point cloud if not present
    if not pcl.has_normals():
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
    
    # Use fewer threads to avoid OpenBLAS issues
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcl, depth=depth, n_threads=1, width=0, scale=1.1, linear_fit=False
        )
    
    # Remove low density vertices (likely outliers)
    if len(densities) > 0:
        vertices_to_remove = densities < np.quantile(densities, 0.0001)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"After removing low density vertices: {len(mesh.vertices)} vertices")
    
    # only keep the largest component
    mesh_trimesh = trimesh.Trimesh(np.array(mesh.vertices), np.array(mesh.triangles))
    return mesh_trimesh

    largest_mesh = keep_largest(mesh_trimesh)
    
    if decimation:
        # mesh decimation for faster rendering
        print("Decimating mesh...")
        low_res_mesh = largest_mesh.simplify_quadric_decimation(50000)
        print(f"Decimated to {len(low_res_mesh.vertices)} vertices")
        return low_res_mesh
    else:
        return largest_mesh
    

class CCHMetrics(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.threshold = cfg.LOSS.CONFIDENCE_THRESHOLD
        self.mask_percentage = cfg.LOSS.CONFIDENCE_MASK_PERCENTAGE

        self.list_of_cfd_values = []
        self.list_of_poisson_cfd_values = []
        self.list_of_poisson_p2s_values = []

        self.filter_by_quantile = True

    def _get_confidence_threshold_from_percentage(self, confidence, image_mask):
        """
        Compute threshold value that masks a certain percentage of foreground pixels with lowest confidence.
        
        Args:
            confidence: Confidence values tensor (B, N, H, W) or any shape
            image_mask: Foreground mask (B, N, H, W) or matching shape, can be boolean or numeric
            
        Returns:
            Threshold value to use for masking (scalar)
        """
        if self.mask_percentage <= 0.0:
            return self.threshold
        
        # Ensure mask is boolean tensor
        if not image_mask.dtype == torch.bool:
            image_mask = image_mask.bool()
        
        # Flatten for easier processing
        confidence_flat = confidence.flatten()
        mask_flat = image_mask.flatten()
        
        # Get confidence values only for foreground pixels
        foreground_conf = confidence_flat[mask_flat]
        
        if foreground_conf.numel() == 0:
            return self.threshold
        
        # Calculate the threshold value for the given percentage
        # We want to mask the lowest mask_percentage of foreground pixels
        # Use quantile to get the threshold (equivalent to percentile)
        # quantile expects value in [0, 1] range, where 0.1 means 10th percentile
        computed_threshold = torch.quantile(foreground_conf.float(), self.mask_percentage)
        
        # Use the computed threshold
        return computed_threshold.item()

    def forward(self, predictions, batch):
        ret = {}

        B, N, H, W, _ = predictions['vc_init'].shape
        K = 5

        if "vc_init_conf" in predictions:
            confidence_raw = predictions['vc_init_conf']
            batch_mask = batch['masks'][:, :N]  # Original mask for valid pixels

            threshold_value = self._get_confidence_threshold_from_percentage(
                confidence_raw, batch_mask
            )
            confidence = confidence_raw > threshold_value
        else:
            confidence = torch.ones_like(predictions['vc_init'])[..., 0].bool()

        assert confidence.shape == batch['masks'][:, :N].shape


        if "vc_init" in predictions and "template_mesh_verts" in batch:
            gt_vc = Pointclouds(
                points=batch['template_mesh_verts']
            )
            pred_vc = predictions['vc_init']
            mask = batch['masks'][:, :N] * confidence 

            pred_vc = rearrange(pred_vc, 'b n h w c -> b (n h w) c')
            mask = rearrange(mask, 'b n h w -> b (n h w)')

            vc_cfd, _, _ = self.masked_metric_cfd(gt_vc, pred_vc, mask)
            ret['vc_cfd'] = vc_cfd

        
        if "vp_init" in predictions and "vp_ptcld" in batch:
            gt_vp = batch['vp_ptcld']
            # gt_vp = check_and_fix_inf_nan(gt_vp, 'gt_vp')
            pred_vp = predictions['vp_init']
            mask = batch['masks'][:, :N] * confidence 

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_cfd, _, _ = self.masked_metric_cfd(gt_vp, pred_vp, mask)
            ret['vp_init_cfd'] = vp_cfd


            # An extra frame in the end 
            # extra_gt_vp = Pointclouds(batch['vp_ptcld'].points_list()[4::K])
            extra_gt_vp = Pointclouds(batch['vp'][4::K])
            extra_pred_vp = rearrange(predictions['vp_init'][:, -1], 'b n h w c -> b (n h w) c')
            mask = rearrange((batch['masks'][:, :N] * confidence), 'b n h w -> b (n h w)')

            extra_vp_cfd, _, _ = self.masked_metric_cfd(extra_gt_vp, extra_pred_vp, mask)
            ret['extra_vp_init_cfd'] = extra_vp_cfd


        if "vp" in predictions and "vp_ptcld" in batch:
            gt_vp = batch['vp_ptcld']
            # gt_vp = check_and_fix_inf_nan(gt_vp, 'gt_vp')
            pred_vp = predictions['vp']
            mask = batch['masks'][:, :N] * confidence 

            pred_vp = rearrange(pred_vp, 'b k n h w c -> (b k) (n h w) c')
            mask = rearrange(mask[:, None].repeat(1, K, 1, 1, 1), 'b k n h w -> (b k) (n h w)')

            vp_cfd, vp_cfd_pred2gt, vp_cfd_gt2pred = self.masked_metric_cfd(gt_vp, pred_vp, mask)
            ret['vp_cfd'] = vp_cfd 
            ret['vp_cfd_pred2gt'] = vp_cfd_pred2gt
            ret['vp_cfd_gt2pred'] = vp_cfd_gt2pred

            self.list_of_cfd_values.append(vp_cfd.item())

            # An extra frame in the end 
            # extra_gt_vp = Pointclouds(batch['vp_ptcld'].points_list()[4::K])
            extra_gt_vp = Pointclouds(batch['vp'][4::K])
            extra_pred_vp = rearrange(predictions['vp'][:, -1], 'b n h w c -> b (n h w) c')
            mask = rearrange((batch['masks'][:, :N] * confidence), 'b n h w -> b (n h w)')

            extra_vp_cfd, _, _ = self.masked_metric_cfd(extra_gt_vp, extra_pred_vp, mask)
            ret['extra_vp_cfd'] = extra_vp_cfd

            # Poisson reconstruction and chamfer distance
            # Batch size is always 1, so squeeze
            extra_pred_vp_squeezed = extra_pred_vp.squeeze(0)  # (n h w, c)
            mask_squeezed = mask.squeeze(0).bool()  # (n h w,)
            
            # Apply mask to get valid points
            valid_points = extra_pred_vp_squeezed[mask_squeezed].cpu().numpy()  # (N, 3)
            print(f"valid_points: {valid_points.shape}")
            
            if len(valid_points) > 0:
                try:
                    # Reconstruct mesh using poisson (pass points directly)
                    reconstructed_mesh = poisson(valid_points, depth=10, decimation=False)
                    # print(f"reconstructed_mesh: {reconstructed_mesh.vertices.shape}")
                    
                    # Convert to pytorch3d Meshes for sampling and chamfer distance
                    mesh_verts = torch.from_numpy(reconstructed_mesh.vertices).float().to(extra_pred_vp.device)
                    mesh_faces = torch.from_numpy(reconstructed_mesh.faces).long().to(extra_pred_vp.device)
                    mesh_pytorch3d = Meshes(verts=[mesh_verts], faces=[mesh_faces])
                    
                    # Sample points from reconstructed mesh
                    num_samples = len(valid_points)  # Sample same number as input points
                    sampled_points_pytorch3d = sample_points_from_meshes(mesh_pytorch3d, 100000)
                    
                    # Compute chamfer distance with extra_gt_vp
                    sampled_points_ptcld = Pointclouds(points=[sampled_points_pytorch3d.squeeze(0)])
                    cfd_ret, _ = chamfer_distance(
                        sampled_points_ptcld, extra_gt_vp,
                        batch_reduction=None,
                        point_reduction=None
                    )
                    
                    cfd_sqrd_pred2gt = cfd_ret[0]
                    cfd_sqrd_gt2pred = cfd_ret[1]
                    
                    cfd_pred2gt = torch.sqrt(cfd_sqrd_pred2gt).mean() * 100.0
                    cfd_gt2pred = torch.sqrt(cfd_sqrd_gt2pred).mean() * 100.0
                    extra_vp_poisson_cfd = (cfd_pred2gt + cfd_gt2pred) / 2
                    
                    # print(f"cfd_pred2gt: {cfd_pred2gt}, cfd_gt2pred: {cfd_gt2pred}", end=' ')
                    print(f"current extra_vp_poisson_cfd: {extra_vp_poisson_cfd.item()}")
                    ret['extra_vp_poisson_cfd'] = extra_vp_poisson_cfd
                    self.list_of_poisson_cfd_values.append(extra_vp_poisson_cfd.item())
                    
                    # Compute point-to-surface distance from reconstructed mesh to GT
                    # Extract GT points (batch size is 1, so get first point cloud)
                    gt_points_list = extra_gt_vp.points_list()
                    gt_points = gt_points_list[0].cpu().numpy()  # (N, 3)
                    
                    # Compute point-to-surface distance using trimesh
                    closest_points, distances, face_indices = reconstructed_mesh.nearest.on_surface(gt_points)
                    distances = np.asarray(distances)
                    
                    # Compute mean distance in centimeters
                    mean_p2s = np.mean(distances) * 100.0
                    
                    print(f"extra_vp_poisson_p2s: {mean_p2s:.4f} cm")
                    self.list_of_poisson_p2s_values.append(mean_p2s)
                except Exception as e:
                    # If reconstruction fails, skip this metric
                    print(f"Warning: Poisson reconstruction failed: {e}")
                    ret['extra_vp_poisson_cfd'] = torch.tensor(float('inf'), device=extra_pred_vp.device)
            else:
                ret['extra_vp_poisson_cfd'] = torch.tensor(float('inf'), device=extra_pred_vp.device)


            print(f"vp_cfd: {vp_cfd.item()}, avg: {np.mean(self.list_of_cfd_values)}")
            print(f"avg of extra_vp_poisson_cfd: {np.mean(self.list_of_poisson_cfd_values)}")
            if len(self.list_of_poisson_p2s_values) > 0:
                print(f"avg of extra_vp_poisson_p2s: {np.mean(self.list_of_poisson_p2s_values):.4f} cm")

            
            
        return ret



    def masked_metric_cfd(self, x_gt, x_pred, mask):

        mask = mask.squeeze(-1).bool()  # (B, V2)

        x_gt_list = x_gt.points_list()
        x_pred_list = [x_pred[b][mask[b]] for b in range(x_pred.shape[0])]

        x_pred_ptclds = Pointclouds(points=x_pred_list)
        
        cfd_ret, _ = chamfer_distance(
            x_pred_ptclds, x_gt, 
            batch_reduction=None, 
            point_reduction=None
        )

        cfd_sqrd_pred2gt = cfd_ret[0]  
        cfd_sqrd_gt2pred = cfd_ret[1]

        # test = torch.sqrt(cfd_sqrd_pred2gt).mean() * 100.0
        # print(f"Test chamfer distance (pred2gt) inside masked_metric_cfd: {test.item()}")
        # test_gt = torch.sqrt(cfd_sqrd_gt2pred).mean() * 100.0
        # print(f"Test chamfer distance (gt2pred) inside masked_metric_cfd: {test_gt.item()}")

        # ---------------------------- pred2gt ----------------------------
        cfd_sqrd_pred2gt_list = []

        for b in range(x_pred.shape[0]):
            cfd_sqrd_b = cfd_sqrd_pred2gt[b][:mask[b].sum()]
            cfd_sqrd_pred2gt_list.append(cfd_sqrd_b)

        cfd_sqrd_pred2gt = torch.cat(cfd_sqrd_pred2gt_list, dim=0)
        cfd_pred2gt = torch.sqrt(cfd_sqrd_pred2gt).mean() * 100.0

        # ---------------------------- gt2pred ----------------------------
        cfd_sqrd_gt2pred_list = []
        for b in range(len(x_gt_list)):
            cfd_sqrd_gt2pred_list.append(cfd_sqrd_gt2pred[b][:x_gt_list[b].shape[0]])
            # No filter for gt2pred 

        cfd_sqrd_gt2pred = torch.cat(cfd_sqrd_gt2pred_list, dim=0)
        cfd_gt2pred = torch.sqrt(cfd_sqrd_gt2pred).mean() * 100.0


        return (cfd_pred2gt+cfd_gt2pred) / 2, cfd_pred2gt, cfd_gt2pred
    
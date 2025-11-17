import os
import torch
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import numpy as np 
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
    
    return  mean_dist


# def compute_chamfer_distance(gt_mesh_path, pred_mesh_path, num_samples=24000, device='cuda'):
#     # Load meshes
#     gt_mesh = trimesh.load(gt_mesh_path, force='mesh')
#     pred_mesh = trimesh.load(pred_mesh_path)
    
#     # Convert to PyTorch3D meshes
#     gt_verts = torch.tensor(gt_mesh.vertices, dtype=torch.float32).to(device)
#     gt_faces = torch.tensor(gt_mesh.faces, dtype=torch.long).to(device)
#     gt_pytorch3d_mesh = Meshes(verts=[gt_verts], faces=[gt_faces])
    
#     pred_verts = torch.tensor(pred_mesh.vertices, dtype=torch.float32).to(device)
    
#     # Sample points from meshes
#     gt_sampled = sample_points_from_meshes(gt_pytorch3d_mesh, num_samples)
#     pred_sampled = pred_verts[None]

    
#     # Compute chamfer distance
#     cfd_ret, _ = chamfer_distance(
#         Pointclouds(points=pred_sampled),
#         Pointclouds(points=gt_sampled),
#         batch_reduction=None,
#         point_reduction=None
#     )
    
#     cfd_sqrd_pred2gt = cfd_ret[0]
#     cfd_sqrd_gt2pred = cfd_ret[1]
    
#     # Convert to centimeters (multiply by 100)
#     cfd_pred2gt = torch.sqrt(cfd_sqrd_pred2gt).mean() * 100.0
#     cfd_gt2pred = torch.sqrt(cfd_sqrd_gt2pred).mean() * 100.0
    
#     return cfd_pred2gt.item(), cfd_gt2pred.item()

def compute_chamfer_distance(gt_mesh_path, pred_mesh_path, num_samples=24000, device='cuda'):

    # Load meshes
    gt_mesh = trimesh.load(gt_mesh_path, force='mesh')
    pred_mesh = trimesh.load(pred_mesh_path, force='mesh')
    
    # Convert to PyTorch3D meshes
    gt_verts = torch.tensor(gt_mesh.vertices, dtype=torch.float32).to(device)
    gt_faces = torch.tensor(gt_mesh.faces, dtype=torch.long).to(device)
    gt_pytorch3d_mesh = Meshes(verts=[gt_verts], faces=[gt_faces])
    
    pred_verts = torch.tensor(pred_mesh.vertices, dtype=torch.float32).to(device)
    pred_faces = torch.tensor(pred_mesh.faces, dtype=torch.long).to(device)
    pred_pytorch3d_mesh = Meshes(verts=[pred_verts], faces=[pred_faces])
    
    # Sample points from meshes
    gt_sampled = sample_points_from_meshes(gt_pytorch3d_mesh, num_samples)
    pred_sampled = sample_points_from_meshes(pred_pytorch3d_mesh, num_samples)
    # gt_sampled = gt_verts[None]
    # pred_sampled = pred_verts[None]
    
    dist = compute_point_to_surface_distance(pred_sampled.squeeze().cpu().numpy(), gt_mesh)
    
    return dist

if __name__ == "__main__":
    # Base directory containing all outputs
    base_dir = "exp/exp_100_5_vp/vis/for_visuals"
    poisson_dir = "exp/exp_100_5_vp/vis/poisson_reconstruction"
    # poisson_dir=base_dir 

    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found at {base_dir}")
        exit(1)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Look for mesh files directly in the base directory


    pred_fnames = sorted([f for f in os.listdir(poisson_dir) if f.startswith('pred_vp_') and not f.startswith('pred_vp_init_')])
    gt_fnames = sorted([f for f in os.listdir(base_dir) if f.startswith('gt_vp_')])
    

    
    print("\n" + "="*70)
    print(f"Processing: {base_dir}")
    print("="*70)

    dist_list = []

    for i in range(len(pred_fnames)):
        gt_mesh_path = os.path.join(base_dir, gt_fnames[i])
        pred_mesh_path = os.path.join(poisson_dir, pred_fnames[i])
        
        try:
            # Compute chamfer distance
            avg_dist = compute_chamfer_distance(
                gt_mesh_path=gt_mesh_path,
                pred_mesh_path=pred_mesh_path,
                num_samples=100000,
                device=device
            )
            avg_dist = avg_dist * 100.0  # convert to cm

            dist_list.append(avg_dist)

            
            print(f"\nResults for {base_dir}:")
            print(f"  Average: {avg_dist:.4f} cm")
            
            
        except Exception as e:
            print(f"Error processing {base_dir}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)    
    if dist_list:
        overall_avg_dist = sum(dist_list) / len(dist_list)
        print(f"Overall Average Distance: {overall_avg_dist:.4f} cm")
    else:
        print("No distances were computed.")    
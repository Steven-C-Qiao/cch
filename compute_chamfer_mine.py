import os
import torch
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

def compute_chamfer_distance(gt_mesh_path, pred_mesh_path, num_samples=24000, device='cuda'):
    # Load meshes
    gt_mesh = trimesh.load(gt_mesh_path, force='mesh')
    pred_mesh = trimesh.load(pred_mesh_path)
    
    # Convert to PyTorch3D meshes
    gt_verts = torch.tensor(gt_mesh.vertices, dtype=torch.float32).to(device)
    gt_faces = torch.tensor(gt_mesh.faces, dtype=torch.long).to(device)
    gt_pytorch3d_mesh = Meshes(verts=[gt_verts], faces=[gt_faces])
    
    pred_verts = torch.tensor(pred_mesh.vertices, dtype=torch.float32).to(device)
    
    # Sample points from meshes
    gt_sampled = sample_points_from_meshes(gt_pytorch3d_mesh, num_samples)
    pred_sampled = pred_verts[None]
    

    
    # Compute chamfer distance
    cfd_ret, _ = chamfer_distance(
        Pointclouds(points=pred_sampled),
        Pointclouds(points=gt_sampled),
        batch_reduction=None,
        point_reduction=None
    )
    
    cfd_sqrd_pred2gt = cfd_ret[0]
    cfd_sqrd_gt2pred = cfd_ret[1]
    
    # Convert to centimeters (multiply by 100)
    cfd_pred2gt = torch.sqrt(cfd_sqrd_pred2gt).mean() * 100.0
    cfd_gt2pred = torch.sqrt(cfd_sqrd_gt2pred).mean() * 100.0
    
    return cfd_pred2gt.item(), cfd_gt2pred.item()

# def compute_chamfer_distance(gt_mesh_path, pred_mesh_path, num_samples=24000, device='cuda'):
#     """
#     Load OBJ meshes, sample them, and compute chamfer distance.
    
#     Args:
#         gt_mesh_path: Path to ground truth mesh OBJ file
#         pred_mesh_path: Path to predicted mesh OBJ file
#         num_samples: Number of points to sample from each mesh
#         device: Device to use for computation ('cuda' or 'cpu')
    
#     Returns:
#         Tuple of (chamfer_distance_pred2gt, chamfer_distance_gt2pred)
#     """
#     # Load meshes
#     gt_mesh = trimesh.load(gt_mesh_path, force='mesh')
#     pred_mesh = trimesh.load(pred_mesh_path, force='mesh')
    
#     # Convert to PyTorch3D meshes
#     gt_verts = torch.tensor(gt_mesh.vertices, dtype=torch.float32).to(device)
#     gt_faces = torch.tensor(gt_mesh.faces, dtype=torch.long).to(device)
#     gt_pytorch3d_mesh = Meshes(verts=[gt_verts], faces=[gt_faces])
    
#     pred_verts = torch.tensor(pred_mesh.vertices, dtype=torch.float32).to(device)
#     pred_faces = torch.tensor(pred_mesh.faces, dtype=torch.long).to(device)
#     pred_pytorch3d_mesh = Meshes(verts=[pred_verts], faces=[pred_faces])
    
#     # Sample points from meshes
#     gt_sampled = sample_points_from_meshes(gt_pytorch3d_mesh, num_samples)
#     pred_sampled = sample_points_from_meshes(pred_pytorch3d_mesh, num_samples)
#     # gt_sampled = gt_verts[None]
#     # pred_sampled = pred_verts[None]
    
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

if __name__ == "__main__":
    # Base directory containing all outputs
    base_dir = "exp/exp_eval/vis/novel_pose_point_clouds"
    poisson_dir = "exp/exp_100_5_vp/vis/poisson_reconstruction"
    poisson_dir=base_dir 

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
    pred2gt_list = []
    gt2pred_list = []

    for i in range(len(pred_fnames)):
        gt_mesh_path = os.path.join(base_dir, gt_fnames[i])
        pred_mesh_path = os.path.join(poisson_dir, pred_fnames[i])
        
        try:
            # Compute chamfer distance
            cfd_pred2gt, cfd_gt2pred = compute_chamfer_distance(
                gt_mesh_path=gt_mesh_path,
                pred_mesh_path=pred_mesh_path,
                num_samples=120000,
                device=device
            )

            
            avg_cfd = (cfd_pred2gt + cfd_gt2pred) / 2.0
            dist_list.append(avg_cfd)
            pred2gt_list.append(cfd_pred2gt)
            gt2pred_list.append(cfd_gt2pred)
            
            print(f"\nResults for {base_dir}:")
            print(f"  Pred -> GT: {cfd_pred2gt:.4f} cm")
            print(f"  GT -> Pred: {cfd_gt2pred:.4f} cm")
            print(f"  Average: {avg_cfd:.4f} cm")
            
            # Print summary
            print("\n" + "="*70)
            print("SUMMARY - Chamfer Distance Results")
            print("="*70)
            print(f"{'Pred->GT (cm)':<15} {'GT->Pred (cm)':<15} {'Average (cm)':<15}")
            print("-"*70)
            print(f"{cfd_pred2gt:<15.4f} {cfd_gt2pred:<15.4f} {avg_cfd:<15.4f}")
            print("="*70)
            
        except Exception as e:
            print(f"Error processing {base_dir}: {e}")
            import traceback
            traceback.print_exc()
    print("\n" + "="*70)
    if dist_list:
        overall_avg = sum(dist_list) / len(dist_list)
        overall_pred2gt = sum(pred2gt_list) / len(pred2gt_list)
        overall_gt2pred = sum(gt2pred_list) / len(gt2pred_list)
        print(f"Overall Average Chamfer Distance across all samples: {overall_avg:.4f} cm")
        print(f"Overall Pred->GT Chamfer Distance across all samples: {overall_pred2gt:.4f} cm")
        print(f"Overall GT->Pred Chamfer Distance across all samples: {overall_gt2pred:.4f} cm")

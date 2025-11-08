import os
import torch
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

def compute_chamfer_distance(gt_mesh_path, pred_mesh_path, num_samples=24000, device='cuda'):
    """
    Load OBJ meshes, sample them, and compute chamfer distance.
    
    Args:
        gt_mesh_path: Path to ground truth mesh OBJ file
        pred_mesh_path: Path to predicted mesh OBJ file
        num_samples: Number of points to sample from each mesh
        device: Device to use for computation ('cuda' or 'cpu')
    
    Returns:
        Tuple of (chamfer_distance_pred2gt, chamfer_distance_gt2pred)
    """
    # Load meshes
    gt_mesh = trimesh.load(gt_mesh_path)
    pred_mesh = trimesh.load(pred_mesh_path)
    
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

if __name__ == "__main__":
    # Base directory containing all outputs
    base_dir = "vis/00134_take1"
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found at {base_dir}")
        exit(1)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Look for mesh files directly in the base directory
    gt_mesh_path = os.path.join(base_dir, "gt_mesh_aligned.obj")
    pred_mesh_path = os.path.join(base_dir, "pred_mesh_aligned.obj")
    
    # Check if files exist
    if not os.path.exists(gt_mesh_path):
        print(f"Warning: GT mesh file not found at {gt_mesh_path}")
        print(f"Available OBJ files in {base_dir}:")
        obj_files = [f for f in os.listdir(base_dir) if f.endswith('.obj')]
        for f in obj_files:
            print(f"  - {f}")
        exit(1)
    
    if not os.path.exists(pred_mesh_path):
        print(f"Warning: Predicted mesh file not found at {pred_mesh_path}")
        print(f"Available OBJ files in {base_dir}:")
        obj_files = [f for f in os.listdir(base_dir) if f.endswith('.obj')]
        for f in obj_files:
            print(f"  - {f}")
        exit(1)
    
    print("\n" + "="*70)
    print(f"Processing: {base_dir}")
    print("="*70)
    
    try:
        # Compute chamfer distance
        cfd_pred2gt, cfd_gt2pred = compute_chamfer_distance(
            gt_mesh_path=gt_mesh_path,
            pred_mesh_path=pred_mesh_path,
            num_samples=200000,
            device=device
        )
        
        avg_cfd = (cfd_pred2gt + cfd_gt2pred) / 2.0
        
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


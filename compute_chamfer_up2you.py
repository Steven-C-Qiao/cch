import os
import torch
import trimesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

def compute_chamfer_distance(gt_mesh_path, pred_mesh_path, num_samples=24000, device='cuda'):
    """
    Load OBJ meshes, scale them to 1.7m, sample them, and compute chamfer distance.
    
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
    
    # Scale both meshes to 1.7m height
    target_height = 1.7  # meters
    
    # Calculate bounding box height for GT mesh
    gt_bbox = gt_mesh.bounds
    gt_height = gt_bbox[1][1] - gt_bbox[0][1]  # Y-axis extent
    gt_scale = target_height / gt_height if gt_height > 0 else 1.0
    gt_mesh.apply_scale(gt_scale)
    
    # Calculate bounding box height for pred mesh
    pred_bbox = pred_mesh.bounds
    pred_height = pred_bbox[1][1] - pred_bbox[0][1]  # Y-axis extent
    pred_scale = target_height / pred_height if pred_height > 0 else 1.0
    pred_mesh.apply_scale(pred_scale)
    
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
    base_dir = "up2you_vis/00134_take1"
    
    # Find all outputs directories
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found at {base_dir}")
        exit(1)
    
    # Get all outputs_* directories and sort them
    all_dirs = [d for d in os.listdir(base_dir) if d.startswith("outputs_") and os.path.isdir(os.path.join(base_dir, d))]
    all_dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else float('inf'))
    
    print(f"Found {len(all_dirs)} outputs directories")
    print(f"Directories: {all_dirs}\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Store results for all directories
    results = []
    
    # Process each outputs directory
    for output_dir in all_dirs:
        mesh_dir = os.path.join(base_dir, output_dir, "meshes")
        gt_mesh_path = os.path.join(mesh_dir, "gt_mesh_aligned.obj")
        pred_mesh_path = os.path.join(mesh_dir, "pred_mesh_aligned.obj")
        
        # Check if files exist
        if not os.path.exists(gt_mesh_path):
            print(f"Warning: GT mesh file not found at {gt_mesh_path}, skipping...")
            continue
        
        if not os.path.exists(pred_mesh_path):
            print(f"Warning: Predicted mesh file not found at {pred_mesh_path}, skipping...")
            continue
        
        print("\n" + "="*70)
        print(f"Processing: {output_dir}")
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
            
            results.append({
                'directory': output_dir,
                'pred2gt': cfd_pred2gt,
                'gt2pred': cfd_gt2pred,
                'average': avg_cfd
            })
            
            print(f"\nResults for {output_dir}:")
            print(f"  Pred -> GT: {cfd_pred2gt:.4f} cm")
            print(f"  GT -> Pred: {cfd_gt2pred:.4f} cm")
            print(f"  Average: {avg_cfd:.4f} cm")
            
        except Exception as e:
            print(f"Error processing {output_dir}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - Chamfer Distance Results for All Directories")
    print("="*70)
    print(f"{'Directory':<20} {'Pred->GT (cm)':<15} {'GT->Pred (cm)':<15} {'Average (cm)':<15}")
    print("-"*70)
    
    for result in results:
        print(f"{result['directory']:<20} {result['pred2gt']:<15.4f} {result['gt2pred']:<15.4f} {result['average']:<15.4f}")
    
    if results:
        avg_pred2gt = sum(r['pred2gt'] for r in results) / len(results)
        avg_gt2pred = sum(r['gt2pred'] for r in results) / len(results)
        avg_overall = sum(r['average'] for r in results) / len(results)
        
        print("-"*70)
        print(f"{'OVERALL AVERAGE':<20} {avg_pred2gt:<15.4f} {avg_gt2pred:<15.4f} {avg_overall:<15.4f}")
        print("="*70)
    else:
        print("No results to display.")


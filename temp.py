import os
import trimesh 
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss import chamfer_distance
import torch

from pytorch3d.ops import sample_points_from_meshes


if __name__ == '__main__':
    # base_dir = 'Figures/vs_up2you/Take10_00117'
    # base_dir = 'Figures/vs_up2you/Take99_00191_14'
    base_dir = 'Figures/vs_up2you/Take1_00079'
    os.makedirs(os.path.join(base_dir, 'scaled'), exist_ok=True)
    gt = os.path.join(base_dir, 'gt.ply')
    pred = os.path.join(base_dir, 'ours.obj')
    up2you = os.path.join(base_dir, 'up2you.obj')
    # up2you_gt = os.path.join(base_dir, 'outputs_14/meshes/gt_mesh_aligned.obj')

    gt_mesh = trimesh.load(gt)
    pred_mesh = trimesh.load(pred)
    # up2you_gt_mesh = trimesh.load(up2you_gt)
    up2you_mesh = trimesh.load(up2you)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_height = gt_mesh.bounds[1][1] - gt_mesh.bounds[0][1]
    pred_height = pred_mesh.bounds[1][1] - pred_mesh.bounds[0][1]
    # up2you_gt_height = up2you_gt_mesh.bounds[1][1] - up2you_gt_mesh.bounds[0][1]
    up2you_height = up2you_mesh.bounds[1][1] - up2you_mesh.bounds[0][1]

    print(gt_height, pred_height, up2you_height)

    gt_scale = 1.0  # Don't scale GT, keep as reference
    pred_scale = gt_height / pred_height  # Scale pred to match GT's height
    up2you_scale = gt_height / up2you_height  # Scale up2you to match GT's height

    gt_mesh.apply_scale(gt_scale)
    pred_mesh.apply_scale(pred_scale)
    # up2you_gt_mesh.apply_scale(up2you_gt_scale)
    up2you_mesh.apply_scale(up2you_scale)

    gt_mesh.export(os.path.join(base_dir, 'scaled', 'gt_scaled.obj'))
    pred_mesh.export(os.path.join(base_dir, 'scaled', 'ours_scaled.obj'))
    # up2you_gt_mesh.export(os.path.join(base_dir, 'scaled', 'up2you_gt_scaled.obj'))
    up2you_mesh.export(os.path.join(base_dir, 'scaled', 'up2you_scaled.obj'))

    gt_verts = torch.tensor(gt_mesh.vertices, dtype=torch.float32).to(device)
    gt_faces = torch.tensor(gt_mesh.faces, dtype=torch.long).to(device)
    gt_pytorch3d_mesh = Meshes(verts=[gt_verts], faces=[gt_faces])

    pred_verts = torch.tensor(pred_mesh.vertices, dtype=torch.float32).to(device)
    pred_faces = torch.tensor(pred_mesh.faces, dtype=torch.long).to(device)
    pred_pytorch3d_mesh = Meshes(verts=[pred_verts], faces=[pred_faces])

    # up2you_gt_verts = torch.tensor(up2you_gt_mesh.vertices, dtype=torch.float32).to(device)
    # up2you_gt_faces = torch.tensor(up2you_gt_mesh.faces, dtype=torch.long).to(device)
    # up2you_gt_pytorch3d_mesh = Meshes(verts=[up2you_gt_verts], faces=[up2you_gt_faces])

    up2you_verts = torch.tensor(up2you_mesh.vertices, dtype=torch.float32).to(device)
    up2you_faces = torch.tensor(up2you_mesh.faces, dtype=torch.long).to(device)
    up2you_pytorch3d_mesh = Meshes(verts=[up2you_verts], faces=[up2you_faces])

    num_samples = 120000
    gt_sampled = sample_points_from_meshes(gt_pytorch3d_mesh, num_samples)
    pred_sampled = sample_points_from_meshes(pred_pytorch3d_mesh, num_samples)
    # up2you_gt_sampled = sample_points_from_meshes(up2you_gt_pytorch3d_mesh, num_samples)
    up2you_sampled = sample_points_from_meshes(up2you_pytorch3d_mesh, num_samples)

    cfd_output, _ = chamfer_distance(
        Pointclouds(points=gt_sampled),
        Pointclouds(points=pred_sampled),
        batch_reduction=None,
        point_reduction=None
    )
    cfd_pred2gt = torch.sqrt(cfd_output[0]).mean() * 100.0
    cfd_gt2pred = torch.sqrt(cfd_output[1]).mean() * 100.0

    print(cfd_pred2gt, cfd_gt2pred)
    print((cfd_gt2pred + cfd_pred2gt) / 2)



    up2you_cfd_output, _ = chamfer_distance(
        Pointclouds(points=gt_sampled),
        Pointclouds(points=up2you_sampled),
        batch_reduction=None,
        point_reduction=None
    )
    up2you_cfd_pred2gt = torch.sqrt(up2you_cfd_output[0]).mean() * 100.0
    up2you_cfd_gt2pred = torch.sqrt(up2you_cfd_output[1]).mean() * 100.0

    print(up2you_cfd_pred2gt, up2you_cfd_gt2pred)
    print((up2you_cfd_gt2pred + up2you_cfd_pred2gt) / 2)
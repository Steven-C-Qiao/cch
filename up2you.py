import pickle 
import trimesh
import torch 

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds
import torch

gt_path = '/scratches/kyuban/cq244/datasets/4DDress/00134/Inner/Take1/Meshes_pkl/mesh-f00102.pkl'

pred_path = '/scratch/cq244/cch/mesh_remeshed.obj'


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(gt_path, 'rb') as f:
        gt_mesh = pickle.load(f)

    pred_mesh_trimesh = trimesh.load(pred_path)

    gt_verts = torch.tensor(gt_mesh['vertices']).float().to(device)
    pred_verts = torch.tensor(pred_mesh_trimesh.vertices).float().to(device)


    # Scale to 1.8m height
    gt_height = gt_verts[:, 1].max() - gt_verts[:, 1].min()
    pred_height = pred_verts[:, 1].max() - pred_verts[:, 1].min()
    
    # Scale to same height
    gt_verts = gt_verts * (1.8 / gt_height)
    pred_verts = pred_verts * (1.8 / pred_height)

    # Center around origin using min-max bounds
    gt_center = (gt_verts.max(dim=0)[0] + gt_verts.min(dim=0)[0]) / 2
    pred_center = (pred_verts.max(dim=0)[0] + pred_verts.min(dim=0)[0]) / 2
    gt_verts = gt_verts - gt_center.unsqueeze(0)
    pred_verts = pred_verts - pred_center.unsqueeze(0)

    




    # Visualize the point clouds in 3D
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Convert to numpy for matplotlib
    gt_verts_np = gt_verts.cpu().numpy()
    pred_verts_np = pred_verts.cpu().numpy()

    fig = plt.figure(figsize=(18, 6))
    
    # Ground truth plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(gt_verts_np[:, 0], gt_verts_np[:, 2], gt_verts_np[:, 1], c='blue', marker='.', alpha=0.6, s=0.3)
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z') 
    ax1.set_zlabel('Y')
    
    # Prediction plot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred_verts_np[:, 0], pred_verts_np[:, 2], pred_verts_np[:, 1], c='red', marker='.', alpha=0.6, s=0.3)
    ax2.set_title('Prediction')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_zlabel('Y')

    # Overlay plot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(gt_verts_np[:, 0], gt_verts_np[:, 2], gt_verts_np[:, 1], c='blue', marker='.', alpha=0.3, s=0.3, label='Ground Truth')
    ax3.scatter(pred_verts_np[:, 0], pred_verts_np[:, 2], pred_verts_np[:, 1], c='red', marker='.', alpha=0.3, s=0.3, label='Prediction')
    ax3.set_title('Overlay')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_zlabel('Y')
    ax3.legend()

    # Set same scale for all plots
    max_range = max(
        max(gt_verts_np.max(0) - gt_verts_np.min(0)),
        max(pred_verts_np.max(0) - pred_verts_np.min(0))
    )
    for ax in [ax1, ax2, ax3]:
        mid_x = (gt_verts_np[:, 0].max() + gt_verts_np[:, 0].min()) * 0.5
        mid_y = (gt_verts_np[:, 2].max() + gt_verts_np[:, 2].min()) * 0.5
        mid_z = (gt_verts_np[:, 1].max() + gt_verts_np[:, 1].min()) * 0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        ax.view_init(elev=10, azim=20, vertical_axis='z')

    plt.tight_layout()
    # plt.show()
    plt.savefig('gt_pred.png')
    plt.close()

    print(gt_verts.shape, pred_verts.shape)




    cfd_ret, _ = chamfer_distance(
        Pointclouds(points=pred_verts[None].to(device)), 
        Pointclouds(points=gt_verts[None].to(device)), 
        batch_reduction=None, 
        point_reduction=None
    )

    cfd_sqrd_pred2gt = cfd_ret[0]  
    cfd_sqrd_gt2pred = cfd_ret[1]

    cfd_pred2gt = torch.sqrt(cfd_sqrd_pred2gt).mean() * 100.0
    cfd_gt2pred = torch.sqrt(cfd_sqrd_gt2pred).mean() * 100.0

    print(cfd_pred2gt, cfd_gt2pred)
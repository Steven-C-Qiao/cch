import torch
import numpy as np


import sys 
sys.path.append('.')
from core.utils.general_lbs import general_lbs





if __name__ == '__main__':
    import numpy as np
    import pickle
    
    import sys 
    sys.path.append('.')
    
    from core.models.smpl import SMPL
    
    smpl_model = SMPL(
        model_path='model_files/smpl',
        num_betas=10,
        gender='neutral'
    )
    parents = smpl_model.parents
    lbs_weights = smpl_model.lbs_weights


    # BETA_PATH = '/scratches/kyuban/cq244/datasets/CAPE/minimal_body_shape/00134/00134_param.pkl'
    BETA_PATH = '/scratches/kyuban/cq244/datasets/CAPE/minimal_body_shape/00032/00032_param.pkl'
    
    with open(BETA_PATH, 'rb') as f:
        beta_data = pickle.load(f)
    shape = torch.from_numpy(beta_data['betas']).float()[None]


    # CAPE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00134/longlong_athletics_trial1/longlong_athletics_trial1.000150.npz'
    CAPE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000150.npz'
    data = np.load(CAPE_PATH)

    for k, v in data.items():
        print(k, v.shape)

    pose = torch.from_numpy(data['pose']).float()[None]
    body_pose, global_orient = pose[:, 3:], pose[:, :3]
    transl = torch.from_numpy(data['transl'])

    v_posed = torch.from_numpy(data['v_posed']).float()[None]
    v_posed = v_posed - transl[None, None, :]

    v_cano = torch.from_numpy(data['v_cano']).float()[None]
    clothed_height = (v_cano[:, :, 1].max(dim=-1).values - v_cano[:, :, 1].min(dim=-1).values)


    template_verts = smpl_model(body_pose=torch.zeros_like(body_pose),
                                betas=shape,
                                global_orient=torch.zeros_like(global_orient))
    naked_cano_verts = template_verts.vertices#.cpu().detach().numpy()
    template_J = template_verts.joints[:, :24]
    naked_height = (naked_cano_verts[:, :, 1].max(dim=-1).values - naked_cano_verts[:, :, 1].min(dim=-1).values)

    # correct template_J and naked_cano_verts by scaling height
    template_J = template_J * (clothed_height[:, None, None] / naked_height[:, None, None])
    naked_cano_verts = naked_cano_verts * (clothed_height[:, None, None] / naked_height[:, None, None])

    # given v_cano, general_lbs should pose v_cano back to v_posed
    # lbs_v_posed, lbs_J_posed = general_lbs(pose, template_J, v_cano, lbs_weights[None], parents)
    lbs_v_posed, lbs_J_posed = general_lbs(pose, template_J, naked_cano_verts, lbs_weights[None], parents)


    naked_cano_verts = naked_cano_verts.cpu().detach().numpy()
    diff = torch.norm(v_posed - lbs_v_posed, dim=-1).squeeze()
    print(diff.abs().max())






    import matplotlib.pyplot as plt
    verts_posed = lbs_v_posed.detach().numpy()
    verts = v_posed.detach().numpy()
    v_cano = v_cano.cpu().detach().numpy()
    # Create 3D scatter plot comparing verts_posed and verts
    fig = plt.figure(figsize=(24, 6))

    # Plot verts_posed
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(verts_posed[0,:,0], 
                verts_posed[0,:,1],
                verts_posed[0,:,2],
                c=diff.cpu().detach().numpy(), s=0.5, alpha=0.5)
    ax1.set_title('Vertices from general_lbs')
    ax1.view_init(elev=10, azim=20, vertical_axis='y')
    ax1.set_box_aspect([1,1,1])

    # Plot verts from SMPL
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(verts[0,:,0],
                verts[0,:,1], 
                verts[0,:,2],
                c='red', s=0.5, alpha=0.5)
    ax2.set_title('Vertices from data')
    ax2.view_init(elev=10, azim=20, vertical_axis='y')
    ax2.set_box_aspect([1,1,1])

    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(verts_posed[0,:,0], 
                verts_posed[0,:,1],
                verts_posed[0,:,2],
                c=diff.cpu().detach().numpy(), s=0.5, alpha=0.5)
    ax3.scatter(verts[0,:,0],
                verts[0,:,1], 
                verts[0,:,2],
                c='red', s=0.5, alpha=0.5)
    ax3.view_init(elev=10, azim=20, vertical_axis='y')
    ax3.set_box_aspect([1,1,1])

    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(naked_cano_verts[0,:,0], 
                naked_cano_verts[0,:,1],
                naked_cano_verts[0,:,2],
                c='blue', s=0.5, alpha=0.5)
    ax4.scatter(v_cano[0,:,0], 
                v_cano[0,:,1],
                v_cano[0,:,2],
                c='green', s=0.5, alpha=0.5)
    ax4.view_init(elev=10, azim=20, vertical_axis='y')
    ax4.set_box_aspect([1,1,1])

    # Set equal tick ratios for both plots
    for ax in [ax1, ax2, ax3, ax4]:
        max_range = np.array([
            verts[0,:,0].max() - verts[0,:,0].min(),
            verts[0,:,1].max() - verts[0,:,1].min(), 
            verts[0,:,2].max() - verts[0,:,2].min()
        ]).max() / 2.0
        
        mid_x = (verts[0,:,0].max() + verts[0,:,0].min()) * 0.5
        mid_y = (verts[0,:,1].max() + verts[0,:,1].min()) * 0.5
        mid_z = (verts[0,:,2].max() + verts[0,:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig('tinkering/CAPE_bad_naked_shape_visualisation.png')
    plt.show()

    import ipdb; ipdb.set_trace()

import torch

from typing import NewType, Tuple
Tensor = NewType('Tensor', torch.Tensor)

from smplx.lbs import batch_rodrigues, batch_rigid_transform, blend_shapes, vertices2joints

def general_lbs(
    pose: Tensor,
    J: Tensor,
    vc: Tensor,
    lbs_weights: Tensor,
    parents: Tensor,
    pose2rot: bool = True,
) -> Tuple[Tensor, Tensor]:
    ''' 
    LBS modified from SMPL lbs. 

    Assumes that the following is known:
        - pose: joint rotations
        - J: joint locations in canonical pose 

    Args:
        vc: B x V x 3
            canonical vertices to be skinned
        w: B x V x J
            skinning weights
        

    Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(vc.shape[0], pose.shape[0])
    device, dtype = vc.device, vc.dtype

    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    v_posed = vc 

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = 24
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


# ipdb> rot_mats.shape
# torch.Size([2, 24, 3, 3])
# ipdb> J.shape
# torch.Size([2, 24, 3])
# ipdb> parents.shape
# torch.Size([24])
# ipdb> 


if __name__ == '__main__':
    import numpy as np
    import pickle
    
    import sys 
    sys.path.append('.')
    
    from core.models.smpl import SMPL
    
    CAPE_PATH = '/scratches/kyuban/cq244/datasets/CAPE/sequences/00032/longshort_ATUsquat/longshort_ATUsquat.000001.npz'
    data = np.load(CAPE_PATH)
    pose = torch.from_numpy(data['pose']).float()[None]
    body_pose, global_orient = pose[:, 3:], pose[:, :3]
    # shape = torch.from_numpy(data['shape']).float()

    BETA_PATH = '/scratches/kyuban/cq244/datasets/CAPE/minimal_body_shape/00032/00032_param.pkl'
    with open(BETA_PATH, 'rb') as f:
        data = pickle.load(f)
    shape = torch.from_numpy(data['betas']).float()[None]
    
    smpl_model = SMPL(
        model_path='model_files/smpl',
        num_betas=10,
        gender='neutral'
    )
    parents = smpl_model.parents
    lbs_weights = smpl_model.lbs_weights


    print(pose.shape)
    print(shape.shape)
    smpl_output = smpl_model(body_pose=body_pose, 
                             betas=shape,
                             global_orient=global_orient,
                             pose2rot=True)
    J = smpl_output.joints[:, :24]
    verts = smpl_output.vertices


    template_verts = smpl_model(body_pose=torch.zeros_like(body_pose),
                                betas=torch.zeros_like(shape),
                                global_orient=torch.zeros_like(global_orient))
    verts_cano = template_verts.vertices
    template_J = template_verts.joints[:, :24]

    print(pose.shape, J.shape, verts_cano.shape, lbs_weights.shape, parents.shape)


    verts_posed, J_posed = general_lbs(pose, template_J, verts_cano, lbs_weights[None], parents)
    

    import matplotlib.pyplot as plt
    verts_posed = verts_posed.detach().numpy()
    verts = verts.detach().numpy()
    # Create 3D scatter plot comparing verts_posed and verts
    fig = plt.figure(figsize=(12, 6))

    # Plot verts_posed
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(verts_posed[0,:,0], 
                verts_posed[0,:,1],
                verts_posed[0,:,2],
                c='blue', s=0.5, alpha=0.5)
    ax1.set_title('Vertices from general_lbs')
    ax1.view_init(elev=10, azim=20, vertical_axis='y')
    ax1.set_box_aspect([1,1,1])

    # Plot verts from SMPL
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(verts[0,:,0],
                verts[0,:,1], 
                verts[0,:,2],
                c='red', s=0.5, alpha=0.5)
    ax2.set_title('Vertices from SMPL')
    ax2.view_init(elev=10, azim=20, vertical_axis='y')
    ax2.set_box_aspect([1,1,1])

    # Set equal tick ratios for both plots
    for ax in [ax1, ax2]:
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
    plt.show()
    plt.savefig('tinkering/general_lbs.png')

    import ipdb; ipdb.set_trace()
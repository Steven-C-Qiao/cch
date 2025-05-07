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
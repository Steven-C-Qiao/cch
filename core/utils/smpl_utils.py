import torch


def get_smplx_full_pose(smpl_model, batch, use_pca=True):
    global_orient = batch['global_orient']
    body_pose = batch['body_pose']
    betas = batch['betas']
    left_hand_pose = batch['left_hand_pose']
    right_hand_pose = batch['right_hand_pose']
    jaw_pose = batch['jaw_pose']
    leye_pose = batch['leye_pose']
    reye_pose = batch['reye_pose']

    if use_pca:
        left_hand_pose = torch.einsum(
            'bi,ij->bj', [left_hand_pose, smpl_model.left_hand_components])
        right_hand_pose = torch.einsum(
            'bi,ij->bj', [right_hand_pose, smpl_model.right_hand_components])

    full_pose = torch.cat([global_orient, body_pose,
                            jaw_pose, leye_pose, reye_pose,
                            left_hand_pose,
                            right_hand_pose], dim=1)
    return full_pose

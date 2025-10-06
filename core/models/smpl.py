import os
import torch
import numpy as np
import contextlib

from typing import Optional, NewType, Tuple
Tensor = NewType('Tensor', torch.Tensor)

from smplx import SMPL as _SMPL
try:
    from smplx.body_models import ModelOutput as SMPLOutput
except ImportError:
    from smplx.utils import SMPLOutput

from smplx.lbs import vertices2joints, lbs

from core.utils.tweaked_lbs import tweaked_lbs

from core.configs import paths

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to 
            - support more joints 
            - support direct mesh input
    """
    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load(paths.J_REGRESSOR_EXTRA)
        J_regressor_cocoplus = np.load(paths.COCOPLUS_REGRESSOR)
        J_regressor_h36m = np.load(paths.H36M_REGRESSOR)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra,
                                                               dtype=torch.float32))
        self.register_buffer('J_regressor_cocoplus', torch.tensor(J_regressor_cocoplus,
                                                                  dtype=torch.float32))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m,
                                                              dtype=torch.float32))
       
    def forward(
        self,
        vc: Optional[Tensor] = None,
        betas: Optional[Tensor] = None,
        body_pose: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts=True,
        return_full_pose: bool = False,
        pose2rot: bool = True,
        **kwargs
    ) -> SMPLOutput:
        """
        Forward pass for the SMPL model
        
        Add support for canonical vertices vc as input. 
        When given, overrides v_template, and assert betas are None
        """
        kwargs['get_skin'] = True
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        if vc is not None:
            assert betas is None, "Canonical vertices vc provided, so betas should be None"
            vertices, joints = tweaked_lbs(
                                torch.zeros_like(betas), # Discard shape since canonical mesh is given
                                full_pose, vc,
                                self.shapedirs, self.posedirs,
                                self.J_regressor, self.parents,
                                self.lbs_weights, pose2rot=pose2rot)
        else: # Use original lbs if no canonical vertices are given
            vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        extra_joints = vertices2joints(self.J_regressor_extra, vertices)
        cocoplus_joints = vertices2joints(self.J_regressor_cocoplus, vertices)
        h36m_joints = vertices2joints(self.J_regressor_h36m, vertices)
        all_joints = torch.cat([joints, extra_joints, cocoplus_joints,
                                h36m_joints], dim=1)

        output = SMPLOutput(vertices=vertices,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=all_joints,
                            betas=betas,
                            full_pose=full_pose)

        return output

        


    def query(self, hmr_output):
        pred_rotmat = hmr_output['pred_rotmat']
        pred_shape = hmr_output['pred_shape']

        smpl_out = self(global_orient=pred_rotmat[:, [0]],
                        body_pose = pred_rotmat[:, 1:],
                        betas = pred_shape,
                        pose2rot=False)
        return smpl_out


 
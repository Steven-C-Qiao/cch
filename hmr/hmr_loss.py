import torch
import torch.nn as nn
import numpy as np



def normalise_joints2D(joints2D, img_wh):
    """
    Normalise 2D joints from [0, img_wh] space to [-1, 1] space.
    """
    return (2.0 * joints2D) / img_wh - 1.0



class HMRLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_targets = ['vertices', 'tpose_vertices']



        self.log_vars = nn.ParameterDict()
        self.loss_fns = {}
        for loss_target in self.loss_targets:
            self.loss_fns[loss_target] = nn.MSELoss(reduction='mean')

    # def forward(self, pred_dict, target_dict, img_wh, normalise=True):

    #     total_loss = 0.
    #     loss_dict = {}

    #     for loss_target in self.loss_targets:
    #         if loss_target == 'joints2D':
    #             joints2D_visib = target_dict['joints2D_visib'].long()
    #             # Selecting only visible 2D joints for loss
    #             pred = pred_dict['joints2D'][joints2D_visib, :]
    #             target = target_dict['joints2D'][joints2D_visib, :]
    #             # Normalising 2D joints to [-1, 1] x [-1, 1] plane.
    #             pred = normalise_joints2D(pred, img_wh)
    #             target = normalise_joints2D(target, img_wh)

            
    #         elif loss_target == 'vertices' and normalise:
    #             pred = pred_dict['vertices'].view(-1, 6890, 3)
    #             target = target_dict['vertices'].view(-1, 6890, 3)
    #             pred = y_scale_correction(pred, target)
            
    #         elif loss_target == 'tpose_vertices' and normalise:
    #             pred = pred_dict['tpose_vertices'].view(-1, 6890, 3)
    #             target = target_dict['tpose_vertices'].view(-1, 6890, 3)
    #             pred = y_scale_correction(pred, target)
            
    #         else:
    #             pred = pred_dict[loss_target]
    #             target = target_dict[loss_target]

    #         loss = self.loss_fns[loss_target](pred, target)
    #         total_loss += loss
    #         loss_dict[loss_target] = loss

    #     return total_loss, loss_dict
    

    def forward(self, vp, vc, vp_pred, vc_pred):
        loss_dict = {}
        # vp = vp.view(-1, 6890, 3)
        # vc = vc.view(-1, 6890, 3)
        # vp_pred = vp_pred.view(-1, 6890, 3)
        # vc_pred = vc_pred.view(-1, 6890, 3)

        loss1 = self.loss_fns['vertices'](vp_pred, vp)
        loss2 = self.loss_fns['vertices'](vc_pred, vc)
        loss = loss1 + loss2

        loss_dict['vp_loss'] = loss1
        loss_dict['vc_loss'] = loss2
        loss_dict['total_loss'] = loss

        return loss, loss_dict
    

def scale_and_translation_transform_batch_torch(P, T):

    P_mean = torch.mean(P, dim=-2, keepdim=True)
    P_trans = P - P_mean
    P_scale = torch.sqrt(torch.sum(P_trans ** 2, dim=(-2, -1), keepdim=True) / P.shape[-2])
    P_normalised = P_trans / P_scale

    T_mean = torch.mean(T, dim=-2, keepdim=True)
    T_scale = torch.sqrt(torch.sum((T - T_mean) ** 2, dim=(-2, -1), keepdim=True) / T.shape[-2])
    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed

def y_scale_correction(P, T):
    P_scale = torch.sqrt(torch.sum(P ** 2, dim=(-2, -1), keepdim=True) / P.shape[-2])
    P_normalised = P / P_scale

    T_scale = torch.sqrt(torch.sum(T ** 2, dim=(-2, -1), keepdim=True) / T.shape[-2])
    P_transformed = P_normalised * T_scale

    return P_transformed
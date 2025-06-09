import torch
import torch.nn as nn
import torch.nn.functional as F

from smplx.lbs import batch_rodrigues


class HMRHead(nn.Module):
    def __init__(self,
                 additional_mlp_dim=0):
        super().__init__()

        num_betas = 10
        mean_cam_t = torch.tensor([0.0, 0.2, 2.5])

        feats_dim = 512

        self.activation = nn.ELU(inplace=True)

        # Attention-Based Fusion Score MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(feats_dim, feats_dim // 2),
            self.activation,
            nn.Linear(feats_dim // 2, num_betas)
        )

        # Shape MLP (combined)
        self.shape_mlp = nn.Sequential(
            nn.Linear(feats_dim + additional_mlp_dim, feats_dim // 2),
            self.activation,
            nn.Linear(feats_dim // 2, feats_dim // 4),
            self.activation,
            nn.Linear(feats_dim // 4, 1)
        )

        # Pose + Cam MLP (per-input)
        self.num_pose_params = 24 * 3
        self.pose_cam_mlp = nn.Sequential(
            nn.Linear(feats_dim + num_betas, feats_dim // 2),
            self.activation,
            nn.Linear(feats_dim // 2, feats_dim // 4),
            self.activation,
            nn.Linear(feats_dim // 4, self.num_pose_params)
        )

        # self.cam_t_mlp = nn.Sequential(
        #     nn.Linear(feats_dim + additional_mlp_dim, feats_dim // 2),
        #     self.activation,
        #     nn.Linear(feats_dim // 2, feats_dim // 4),
        #     self.activation,
        #     nn.Linear(feats_dim // 4, 3)
        # )
        # self.register_buffer('init_cam_t', mean_cam_t.float())

    def forward(self, feats):
        B, N = feats.shape[:2]


        # Shape fusion scores
        scores = self.score_mlp(feats).transpose(1, 2)  # (B, num_betas, N)
        scores = F.softmax(scores, dim=-1)

        # Shape
        fused_feats = torch.matmul(scores, feats)  # (B, num_betas, D)
        shape = self.shape_mlp(fused_feats)[..., 0]  # (B, num_betas)

        # Pose
        pose_feats = torch.cat([feats, shape[:, None, :].expand(-1, N, -1)], dim=-1)
        pose_cam = self.pose_cam_mlp(pose_feats)  # (B, N, num_pose_params + 3)
        pose = pose_cam[..., :self.num_pose_params]  # (B, N, num_pose_params)
        pose_rotmats = batch_rodrigues(pose.reshape(-1, 3)).view(B, N, 24, 3, 3)

        # Camera translation
        # cam_t = self.init_cam_t + pose_cam[..., self.num_pose_params:]  # (B, N, 3)
        # cam_t = self.cam_t_mlp(feats) + self.init_cam_t  # (B, N, 3)

        pred_dict = {
            'pred_shape': shape,
            'pred_rotmat': pose_rotmats.view(-1, 24, 3, 3),
            # 'pred_cam': cam_t.view(-1, 3)
        }

        return pred_dict
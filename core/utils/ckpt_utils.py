import torch
import torch.nn.functional as F


def interpolate_dino_pos_embed(pos_embed_pretrained, model, num_patches_new):
    """
    Interpolate DINOv2 positional embeddings to match new patch grid size.
    
    Args:
        pos_embed_pretrained: Pretrained positional embeddings
        model: The model containing the new positional embedding
        num_patches_new: Number of patches in the new grid
    
    Returns:
        Interpolated positional embeddings with cls token
    """
    cls_token = pos_embed_pretrained[:, 0:1, :]
    pos_tokens = pos_embed_pretrained[:, 1:, :]

    # Determine the old grid size
    num_patches_old = pos_tokens.shape[1]
    old_size = int(num_patches_old ** 0.5)
    new_size = int(num_patches_new ** 0.5)

    pos_tokens = pos_tokens.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)  # (1, C, H, W)
    pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, -1)

    return torch.cat((cls_token, pos_tokens), dim=1)


def load_and_freeze_pretrained_dinov2(model, ckpt_path='/scratches/kyuban/cq244/CCH/cch/model_files/dinov2_vits14_reg4_pretrain.pth'):
    """
    Load pretrained DINOv2 weights with positional embedding interpolation if needed.
    
    Args:
        model: The model to load weights into
        ckpt_path: Path to the pretrained checkpoint
    
    Returns:
        None (modifies model in-place)
    """
    ckpt = torch.load(ckpt_path, weights_only=False)
    
    # Remove 'pos_embed' temporarily if shape mismatches
    if ckpt["pos_embed"].shape != model.patch_embed.pos_embed.shape:
        print(f"Interpolating pos_embed from {ckpt['pos_embed'].shape} to {model.patch_embed.pos_embed.shape}")
        pos_embed_pretrained = ckpt.pop("pos_embed")

        # Apply interpolation and assign to model
        interpolated_pos_embed = interpolate_dino_pos_embed(pos_embed_pretrained, model, num_patches_new=16*16)
        ckpt["pos_embed"] = interpolated_pos_embed
    else:
        pos_embed_pretrained = None

    # Load all other weights
    msg = model.patch_embed.load_state_dict(ckpt, strict=True)
    # print("Missing keys:", msg.missing_keys)
    # print("Unexpected keys:", msg.unexpected_keys)

    # Freeze all parameters in patch_embed
    for param in model.patch_embed.parameters():
        param.requires_grad = False

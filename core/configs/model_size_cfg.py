MODEL_CONFIGS = {
    'large': {
        'patch_embed': "dinov2_vitl14_reg",
        'img_size': 518,
        'patch_size': 14,
        'embed_dim': 1024,
        'depth': 24,
        'intermediate_layer_idx': [4, 11, 17, 23],
        'pbs_embed_dim': 1024,
        'features': 256,
        'out_channels': [256, 512, 1024, 1024]
    },
    'base': {
        'patch_embed': "dinov2_vitb14_reg",
        'img_size': 224,
        'patch_size': 14,
        'embed_dim': 768,
        'depth': 12,
        'intermediate_layer_idx': [2, 5, 8, 11],
        'pbs_embed_dim': 384,
        'features': 256,
        'out_channels': [256, 512, 1024, 1024]
    },
    'small': {
        'patch_embed': "dinov2_vits14_reg",
        'img_size': 224,
        'patch_size': 14,
        'embed_dim': 384,
        'depth': 12,
        'intermediate_layer_idx': [2, 5, 8, 11],
        'pbs_embed_dim': 384,
        'features': 128,
        'out_channels': [96, 192, 384, 768]
    },
    'tiny': {
        'patch_embed': "dinov2_vits14_reg",
        'img_size': 224,
        'patch_size': 14,
        'embed_dim': 384,
        'depth': 12,
        'intermediate_layer_idx': [2, 5, 8, 11],
        'pbs_embed_dim': 192,
        'features': 64,
        'out_channels': [48, 96, 192, 384]
    },
}
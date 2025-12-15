import torch.nn as nn
from functools import partial


mit_b0 = dict(
    patch_size=4, 
    embed_dims=[32, 64, 160, 256], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[2, 2, 2, 2], 
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, 
    drop_path_rate=0.1
)

mit_b1 = dict(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[2, 2, 2, 2], 
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, 
    drop_path_rate=0.1
)

mit_b2 = dict(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 4, 6, 3], 
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, 
    drop_path_rate=0.1
)

mit_b3 = dict(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 4, 18, 3], 
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, 
    drop_path_rate=0.1
)

mit_b4 = dict(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 8, 27, 3], 
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, 
    drop_path_rate=0.1
)

mit_b5 = dict(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4],
    qkv_bias=True, 
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 6, 40, 3], 
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0, 
    drop_path_rate=0.1
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from utils.engine.logger import get_logger
from .dual_segformer import DWConv, Mlp, Attention, Block, OverlapPatchEmbed

from ..backbone_config.segformer import *

logger = get_logger()


class DoubleMiT(nn.Module):
    def __init__(self, RGBBackbone='mit_b0', DBackbone='mit_b0', norm_fuse=nn.BatchNorm2d, downsample_ratio=0.5):
        super().__init__()
        self.rgb_backbone = RGBBackbone
        self.d_backbone = DBackbone
        self.get_backbone_config()
        logger.info(f"Setting downsample_ratio as {downsample_ratio}")

        self.downsample_ratio = downsample_ratio

        assert sum(self.rgb_model_config['embed_dims']) >= sum(self.d_model_config['embed_dims'])

        # patch embed
        self.rgb_patch_embed_1 = OverlapPatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=3, 
                                                   embed_dim=self.rgb_model_config['embed_dims'][0])
        
        self.rgb_patch_embed_2 = OverlapPatchEmbed(img_size=56, patch_size=3, stride=2, 
                                                   in_chans=self.rgb_model_config['embed_dims'][0],
                                                   embed_dim=self.rgb_model_config['embed_dims'][1])
        
        self.rgb_patch_embed_3 = OverlapPatchEmbed(img_size=28, patch_size=3, stride=2, 
                                                   in_chans=self.rgb_model_config['embed_dims'][1],
                                                   embed_dim=self.rgb_model_config['embed_dims'][2])
        
        self.rgb_patch_embed_4 = OverlapPatchEmbed(img_size=14, patch_size=3, stride=2, 
                                                   in_chans=self.rgb_model_config['embed_dims'][2],
                                                   embed_dim=self.rgb_model_config['embed_dims'][3])
        
        self.d_patch_embed_1 = OverlapPatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=3, 
                                                   embed_dim=self.d_model_config['embed_dims'][0])
        
        self.d_patch_embed_2 = OverlapPatchEmbed(img_size=56, patch_size=3, stride=2, 
                                                   in_chans=self.d_model_config['embed_dims'][0],
                                                   embed_dim=self.d_model_config['embed_dims'][1])
        
        self.d_patch_embed_3 = OverlapPatchEmbed(img_size=28, patch_size=3, stride=2, 
                                                   in_chans=self.d_model_config['embed_dims'][1],
                                                   embed_dim=self.d_model_config['embed_dims'][2])
        
        self.d_patch_embed_4 = OverlapPatchEmbed(img_size=14, patch_size=3, stride=2, 
                                                   in_chans=self.d_model_config['embed_dims'][2],
                                                   embed_dim=self.d_model_config['embed_dims'][3])
        
        # transformer encoder
        self.rgb_dpr = [x.item() for x in torch.linspace(0, self.rgb_model_config['drop_path_rate'], 
                                                    sum(self.rgb_model_config['depths']))]  # stochastic depth decay rule
        self.rgb_cur = 0

        self.rgb_block_1, self.rgb_norm1 = self.get_block(feature_type='rgb', block_num=0)
        self.rgb_cur += self.rgb_model_config['depths'][0]

        self.rgb_block_2, self.rgb_norm2 = self.get_block(feature_type='rgb', block_num=1)
        self.rgb_cur += self.rgb_model_config['depths'][1]

        self.rgb_block_3, self.rgb_norm3 = self.get_block(feature_type='rgb', block_num=2)
        self.rgb_cur += self.rgb_model_config['depths'][2]

        self.rgb_block_4, self.rgb_norm4 = self.get_block(feature_type='rgb', block_num=3)
        self.rgb_cur += self.rgb_model_config['depths'][3]

        self.d_dpr = [x.item() for x in torch.linspace(0, self.d_model_config['drop_path_rate'], 
                                                    sum(self.d_model_config['depths']))]  # stochastic depth decay rule
        self.d_cur = 0

        self.d_block_1, self.d_norm1 = self.get_block(feature_type='d', block_num=0)
        self.d_cur += self.d_model_config['depths'][0]

        self.d_block_2, self.d_norm2 = self.get_block(feature_type='d', block_num=1)
        self.d_cur += self.d_model_config['depths'][1]

        self.d_block_3, self.d_norm3 = self.get_block(feature_type='d', block_num=2)
        self.d_cur += self.d_model_config['depths'][2]

        self.d_block_4, self.d_norm4 = self.get_block(feature_type='d', block_num=3)
        self.d_cur += self.d_model_config['depths'][3]

        self.FRMs = nn.ModuleList([
                    FRM(dim=self.rgb_model_config['embed_dims'][0], reduction=1),
                    FRM(dim=self.rgb_model_config['embed_dims'][1], reduction=1),
                    FRM(dim=self.rgb_model_config['embed_dims'][2], reduction=1),
                    FRM(dim=self.rgb_model_config['embed_dims'][3], reduction=1)])

        self.FFMs = nn.ModuleList([
                    FFM(dim=self.rgb_model_config['embed_dims'][0], reduction=1, 
                        num_heads=self.rgb_model_config['num_heads'][0], norm_layer=norm_fuse),
                    FFM(dim=self.rgb_model_config['embed_dims'][1], reduction=1, 
                        num_heads=self.rgb_model_config['num_heads'][1], norm_layer=norm_fuse),
                    FFM(dim=self.rgb_model_config['embed_dims'][2], reduction=1, 
                        num_heads=self.rgb_model_config['num_heads'][2], norm_layer=norm_fuse),
                    FFM(dim=self.rgb_model_config['embed_dims'][3], reduction=1, 
                        num_heads=self.rgb_model_config['num_heads'][3], norm_layer=norm_fuse)])
        
        self.d_before_layer_1 = nn.Conv2d(in_channels=self.d_model_config['embed_dims'][0],
                                        out_channels=self.rgb_model_config['embed_dims'][0],
                                        kernel_size=1, stride=1)
        self.d_after_layer_1 = nn.Conv2d(in_channels=self.rgb_model_config['embed_dims'][0],
                                       out_channels=self.d_model_config['embed_dims'][0],
                                       kernel_size=1, stride=1)
        
        self.d_before_layer_2 = nn.Conv2d(in_channels=self.d_model_config['embed_dims'][1],
                                        out_channels=self.rgb_model_config['embed_dims'][1],
                                        kernel_size=1, stride=1)
        self.d_after_layer_2 = nn.Conv2d(in_channels=self.rgb_model_config['embed_dims'][1],
                                       out_channels=self.d_model_config['embed_dims'][1],
                                       kernel_size=1, stride=1)
        
        self.d_before_layer_3 = nn.Conv2d(in_channels=self.d_model_config['embed_dims'][2],
                                        out_channels=self.rgb_model_config['embed_dims'][2],
                                        kernel_size=1, stride=1)
        self.d_after_layer_3 = nn.Conv2d(in_channels=self.rgb_model_config['embed_dims'][2],
                                       out_channels=self.d_model_config['embed_dims'][2],
                                       kernel_size=1, stride=1)
        
        self.d_before_layer_4 = nn.Conv2d(in_channels=self.d_model_config['embed_dims'][3],
                                        out_channels=self.rgb_model_config['embed_dims'][3],
                                        kernel_size=1, stride=1)
        # self.d_after_layer_4 = nn.Conv2d(in_channels=self.rgb_model_config['embed_dims'][3],
        #                                out_channels=self.d_model_config['embed_dims'][3],
        #                                kernel_size=1, stride=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained_rgb=None, pretrained_d=None):
        if isinstance(pretrained_rgb, str) and isinstance(pretrained_d, str):
            load_DoubleMiT_model(self, pretrained_rgb, pretrained_d)
        else:
            raise NotImplementedError('pretrained rgb & d must be strs.')

    def get_backbone_config(self):
        # get rgb backbone config
        if self.rgb_backbone == 'mit_b0':
            self.rgb_model_config = mit_b0
        elif self.rgb_backbone == 'mit_b1':
            self.rgb_model_config = mit_b1
        elif self.rgb_backbone == 'mit_b2':
            self.rgb_model_config = mit_b2
        elif self.rgb_backbone == 'mit_b3':
            self.rgb_model_config = mit_b3
        elif self.rgb_backbone == 'mit_b4':
            self.rgb_model_config = mit_b4
        elif self.rgb_backbone == 'mit_b5':
            self.rgb_model_config = mit_b5
        else:
            raise NotImplementedError
        logger.info(f"Using rgb backbone: {self.rgb_backbone}")

        # get depth backbone config
        if self.d_backbone == 'mit_b0':
            self.d_model_config = mit_b0
        elif self.d_backbone == 'mit_b1':
            self.d_model_config = mit_b1
        elif self.d_backbone == 'mit_b2':
            self.d_model_config = mit_b2
        elif self.d_backbone == 'mit_b3':
            self.d_model_config = mit_b3
        elif self.d_backbone == 'mit_b4':
            self.d_model_config = mit_b4
        elif self.d_backbone == 'mit_b5':
            self.d_model_config = mit_b5
        else:
            raise NotImplementedError
        logger.info(f"Using depth backbone: {self.d_backbone}")

    def get_block(self, feature_type='rgb', block_num=0):
        assert block_num in [0, 1, 2, 3]
        if feature_type == 'rgb':
            block = nn.ModuleList([Block(
                dim=self.rgb_model_config['embed_dims'][block_num], 
                num_heads=self.rgb_model_config['num_heads'][block_num], 
                mlp_ratio=self.rgb_model_config['mlp_ratios'][block_num], 
                qkv_bias=self.rgb_model_config['qkv_bias'], 
                qk_scale=None,
                drop=self.rgb_model_config['drop_rate'], 
                attn_drop=0., 
                drop_path=self.rgb_dpr[self.rgb_cur + i], 
                norm_layer=self.rgb_model_config['norm_layer'],
                sr_ratio=self.rgb_model_config['sr_ratios'][block_num]) 
                for i in range(self.rgb_model_config['depths'][block_num])]) 
            norm = self.rgb_model_config['norm_layer'](self.rgb_model_config['embed_dims'][block_num])
        elif feature_type == 'd':
            block = nn.ModuleList([Block(
                dim=self.d_model_config['embed_dims'][block_num], 
                num_heads=self.d_model_config['num_heads'][block_num], 
                mlp_ratio=self.d_model_config['mlp_ratios'][block_num], 
                qkv_bias=self.d_model_config['qkv_bias'], 
                qk_scale=None,
                drop=self.d_model_config['drop_rate'], 
                attn_drop=0., 
                drop_path=self.d_dpr[self.d_cur + i], 
                norm_layer=self.d_model_config['norm_layer'],
                sr_ratio=self.d_model_config['sr_ratios'][block_num]) 
                for i in range(self.d_model_config['depths'][block_num])]) 
            norm = self.d_model_config['norm_layer'](self.d_model_config['embed_dims'][block_num])
        else:
            raise NotImplementedError
        return block, norm
    
    def forward_features(self, x_rgb, x_e):
        B, _, H, W = x_rgb.shape
        outs = []
        outs_fused = []

        h, w = int(H * self.downsample_ratio), int(W * self.downsample_ratio)
        x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=True)

        # stage 1
        x_rgb, h, w = self.rgb_patch_embed_1(x_rgb)

        # B H*W/16 C
        x_e, H, W = self.d_patch_embed_1(x_e)
        for i, blk in enumerate(self.rgb_block_1):
            x_rgb = blk(x_rgb, h, w)
        for i, blk in enumerate(self.d_block_1):
            x_e = blk(x_e, H, W)
        x_rgb = self.rgb_norm1(x_rgb)
        x_e = self.d_norm1(x_e)

        x_rgb = x_rgb.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb = F.interpolate(x_rgb, (H, W), mode='bilinear', align_corners=True)
        x_e = self.d_before_layer_1(x_e)

        x_rgb, x_e = self.FRMs[0](x_rgb, x_e)
        x_fused = self.FFMs[0](x_rgb, x_e)
        outs.append(x_fused)
        
        x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=True)
        x_e = self.d_after_layer_1(x_e)

        # stage 2
        x_rgb, h, w = self.rgb_patch_embed_2(x_rgb)
        x_e, H, W = self.d_patch_embed_2(x_e)
        for i, blk in enumerate(self.rgb_block_2):
            x_rgb = blk(x_rgb, h, w)
        for i, blk in enumerate(self.d_block_2):
            x_e = blk(x_e, H, W)
        x_rgb = self.rgb_norm2(x_rgb)
        x_e = self.d_norm2(x_e)

        x_rgb = x_rgb.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb = F.interpolate(x_rgb, (H, W), mode='bilinear', align_corners=True)
        x_e = self.d_before_layer_2(x_e)

        x_rgb, x_e = self.FRMs[1](x_rgb, x_e)
        x_fused = self.FFMs[1](x_rgb, x_e)
        outs.append(x_fused)
        
        x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=True)
        x_e = self.d_after_layer_2(x_e)

        # stage 3
        x_rgb, h, w = self.rgb_patch_embed_3(x_rgb)
        x_e, H, W = self.d_patch_embed_3(x_e)
        for i, blk in enumerate(self.rgb_block_3):
            x_rgb = blk(x_rgb, h, w)
        for i, blk in enumerate(self.d_block_3):
            x_e = blk(x_e, H, W)
        x_rgb = self.rgb_norm3(x_rgb)
        x_e = self.d_norm3(x_e)

        x_rgb = x_rgb.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb = F.interpolate(x_rgb, (H, W), mode='bilinear', align_corners=True)
        x_e = self.d_before_layer_3(x_e)

        x_rgb, x_e = self.FRMs[2](x_rgb, x_e)
        x_fused = self.FFMs[2](x_rgb, x_e)
        outs.append(x_fused)
        
        x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=True)
        x_e = self.d_after_layer_3(x_e)

        # stage 4
        x_rgb, h, w = self.rgb_patch_embed_4(x_rgb)
        x_e, H, W = self.d_patch_embed_4(x_e)
        for i, blk in enumerate(self.rgb_block_4):
            x_rgb = blk(x_rgb, h, w)
        for i, blk in enumerate(self.d_block_4):
            x_e = blk(x_e, H, W)
        x_rgb = self.rgb_norm4(x_rgb)
        x_e = self.d_norm4(x_e)

        x_rgb = x_rgb.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x_rgb = F.interpolate(x_rgb, (H, W), mode='bilinear', align_corners=True)
        x_e = self.d_before_layer_4(x_e)

        x_rgb, x_e = self.FRMs[3](x_rgb, x_e)
        x_fused = self.FFMs[3](x_rgb, x_e)
        outs.append(x_fused)
        
        return outs
    
    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out
    

def load_DoubleMiT_model(model, rgb_pth, d_pth):
    t_start = time.time()

    if isinstance(rgb_pth, str):
        raw_rgb_state_dict = torch.load(rgb_pth, map_location=torch.device('cpu'))
        if 'model' in raw_rgb_state_dict.keys():
            raw_rgb_state_dict = raw_rgb_state_dict['model']
    else:
        raw_rgb_state_dict = rgb_pth
    if isinstance(d_pth, str):
        raw_d_state_dict = torch.load(d_pth, map_location=torch.device('cpu'))
        if 'model' in raw_d_state_dict.keys():
            raw_d_state_dict = raw_d_state_dict['model']
    else:
        raw_d_state_dict = d_pth

    state_dict = {}
    for k, v in raw_rgb_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k.replace('patch_embed', 'rgb_patch_embed_')] = v
        elif k.find('block') >= 0:
            state_dict[k.replace('block', 'rgb_block_')] = v
        elif k.find('norm') >= 0:
            state_dict[k.replace('norm', 'rgb_norm')] = v
    for k, v in raw_d_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k.replace('patch_embed', 'd_patch_embed_')] = v
        elif k.find('block') >= 0:
            state_dict[k.replace('block', 'd_block_')] = v
        elif k.find('norm') >= 0:
            state_dict[k.replace('norm', 'd_norm')] = v

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))
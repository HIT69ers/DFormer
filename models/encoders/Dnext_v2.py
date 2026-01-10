import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..my_utils.asymformer_module import SCC_Module
import math
import time
from utils.engine.logger import get_logger

from .convnextv2 import *


logger = get_logger()


class DNeXtV2(nn.Module):
    def __init__(self, rgb_backbone="N", d_backbone="A", downsample_ratio=0.5, output_to_depth=True, stage1_scc=False, **kwargs):
        super().__init__()
        self.rgb_backbone = rgb_backbone
        self.d_backbone = d_backbone
        rgb_model, d_model = self._get_model(**kwargs)
        logger.info(f"Using RGB backbone: ConvNeXtV2_{self.rgb_backbone}")
        logger.info(f"Using D backbone: ConvNeXtV2_{self.d_backbone}")
        self.downsample_ratio = downsample_ratio
        logger.info(f"Setting downsample_ratio as {self.downsample_ratio}")

        self.rgb_downsample_layers = rgb_model.downsample_layers
        self.d_downsample_layers = d_model.downsample_layers
        self.rgb_stages = rgb_model.stages
        self.d_stages = d_model.stages

        self.rgb_dims, self.d_dims = self._get_dims()

        # If true, adding feature fusion after stage 1
        self.stage1_scc = stage1_scc
        if not self.stage1_scc:
            self.SCCs = nn.ModuleList(
                [SCC_Module(inc_depth2=self.d_dims[i], inc_rgb=self.rgb_dims[i]) for i in range(1, 4)]
            )
        else:
            self.SCCs = nn.ModuleList(
                [SCC_Module(inc_depth2=self.d_dims[i], inc_rgb=self.rgb_dims[i]) for i in range(4)]
            )

        # If adding fused feature maps into depth backbone
        self.output_to_depth = output_to_depth

    def _init_weights(self, m):
        ## 注意：该初始化方式来自ConvNeXtV2，不同于CMX中对FRM和FFM中参数初始化的方式
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained_rgb=None, pretrained_d=None):
        load_DNeXtV2_model(self, pretrained_rgb, pretrained_d)

    def _get_model(self, **kwargs):
        model_dict = dict(
            A=convnextv2_atto,
            F=convnextv2_femto,
            P=convnext_pico,
            N=convnextv2_nano,
            T=convnextv2_tiny,
            B=convnextv2_base,
            L=convnextv2_large,
            H=convnextv2_huge
        )
        # self.rgb_model = model_dict[self.rgb_backbone](**kwargs)
        # self.d_model = model_dict[self.d_backbone](**kwargs)
        return model_dict[self.rgb_backbone](**kwargs), model_dict[self.d_backbone](**kwargs)
    
    def _get_dims(self):
        model_dims = dict(
            A=[40, 80, 160, 320],
            F=[48, 96, 192, 384],
            P=[64, 128, 256, 512],
            N=[80, 160, 320, 640],
            T=[96, 192, 384, 768],
            B=[128, 256, 512, 1024],
            L=[192, 384, 768, 1536],
            H=[352, 704, 1408, 2816]
        )
        return model_dims[self.rgb_backbone], model_dims[self.d_backbone]
    
    def forward_features(self, x_rgb, x_d):
        outs = []
        # stages 1-4
        for i in range(4):
            # backbone
            x_rgb = self.rgb_downsample_layers[i](x_rgb)
            if i == 0:  # 在stage 1 的下采样后降低rgb特征的分辨率
                B, C, H, W = x_rgb.shape
                h, w = int(H * self.downsample_ratio), int(W * self.downsample_ratio)
                x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=True)
            x_d = self.d_downsample_layers[i](x_d)

            x_rgb = self.rgb_stages[i](x_rgb)
            x_d = self.d_stages[i](x_d)

            # before feature fusion
            _, _, h, w = x_rgb.shape
            _, _, H, W = x_d.shape
            x_rgb = F.interpolate(x_rgb, (H, W), mode='bilinear', align_corners=True)

            # feature fusion
            if not self.stage1_scc:
                if i > 0:
                    x_fused = self.SCCs[i - 1](x_d, x_rgb)
                else:
                    x_fused = x_d
            else:
                x_fused = self.SCCs[i](x_d, x_rgb)
            outs.append(x_fused)

            # after feature fusion
            if i < 3:
                _, _, H, W = x_d.shape
                h, w = int(H * self.downsample_ratio), int(W * self.downsample_ratio)
                x_rgb = F.interpolate(x_rgb, (h, w), mode='bilinear', align_corners=True)

            # Adding fused feature map into depth backbone
            if self.output_to_depth:
                x_d = x_fused

        return outs
    
    def forward(self, x_rgb, x_d):
        return self.forward_features(x_rgb, x_d)
    

def load_DNeXtV2_model(model, rgb_pt, d_pt):
    if isinstance(rgb_pt, str):
        logger.info(f"RGB pretrained backbone: {rgb_pt}")
        raw_rgb_state_dict = torch.load(rgb_pt, map_location=torch.device('cpu'))
        if 'model' in raw_rgb_state_dict.keys():
            raw_rgb_state_dict = raw_rgb_state_dict['model']
    else:
        raw_rgb_state_dict = rgb_pt
    if isinstance(d_pt, str):
        logger.info(f"Depth pretrained backbone: {d_pt}")
        raw_d_state_dict = torch.load(d_pt, map_location=torch.device('cpu'))
        if 'model' in raw_d_state_dict.keys():
            raw_d_state_dict = raw_d_state_dict['model']
    else:
        raw_d_state_dict = d_pt

    state_dict = {}
    if raw_rgb_state_dict is not None:
        for k, v in raw_rgb_state_dict.items():
            if k.find('downsample_layers') >= 0:
                state_dict[k.replace('downsample_layers', 'rgb_downsample_layers')] = v
            elif k.find('stages') >= 0:
                state_dict[k.replace('stages', 'rgb_stages')] = v
    if raw_d_state_dict is not None:
        for k, v in raw_d_state_dict.items():
            if k.find('downsample_layers') >= 0:
                state_dict[k.replace('downsample_layers', 'd_downsample_layers')] = v
            elif k.find('stages') >= 0:
                state_dict[k.replace('stages', 'd_stages')] = v
    
    if (raw_d_state_dict is None) and (raw_rgb_state_dict is None):
        logger.warning(f"Both rgb and depth pretrained weights are None. Please check again.")
    else:
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        logger.info(f"Successfully loading DNeXtV2 model!")

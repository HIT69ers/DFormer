import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from thop import profile


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def test_asym_rgbd():
    from models.encoders.double_mit import DoubleMiT

    rgb_input = torch.randn(1, 3, 224, 224)
    d_input = torch.randn(1, 3, 224, 224)

    asymrgbd = DoubleMiT(RGBBackbone='mit_b1', DBackbone='mit_b0', downsample_ratio=0.7)
    asymrgbd.init_weights(pretrained_rgb="E:\Files\PretrainedModel\Segformer\mit_b1.pth",
                          pretrained_d="E:\Files\PretrainedModel\Segformer\mit_b0.pth")
    outputs = asymrgbd(rgb_input, d_input)
    print(f"output 1 shape: {outputs[0].shape}")
    print(f"output 2 shape: {outputs[1].shape}")
    print(f"output 3 shape: {outputs[2].shape}")
    print(f"output 4 shape: {outputs[3].shape}")


def learn_layernorm():
    B, C, H, W = 2, 3, 4, 5
    x = torch.rand(B, C, H, W).permute(0, 2, 3, 1)
    print(f"x.shape: {x.shape}")
    layernorm = LayerNorm(C, data_format="channels_last")
    y = layernorm(x)
    # print(y.shape)
    # print(y.sum(dim=1))
    # print(y.var(dim=1))
    # a = torch.ones(C)
    # print(a[:, None, None])


def test_double_NeXtV2():
    print(f"-------------------------test_double_NeXtV2-------------------------")
    from models.encoders.double_next_v2 import DoubleNeXtV2

    dummy_rgb, dummy_depth = torch.randn(1, 3, 480, 640), torch.randn(1, 3, 480, 640)

    double_nextv2 = DoubleNeXtV2(rgb_backbone="N", d_backbone="A", downsample_ratio=1.0, drop_path_rate=0.1)
    double_nextv2.init_weights(pretrained_rgb="/mnt/syh/pretrained/ConvNeXtV2/ImageNet-22K/convnextv2_nano_22k_384_ema.pt",
                               pretrained_d="/mnt/syh/pretrained/ConvNeXtV2/ImageNet-1K/convnextv2_atto_1k_224_ema.pt")
    outputs = double_nextv2(dummy_rgb, dummy_depth)
    print(f"output 1 shape: {outputs[0].shape}")
    print(f"output 2 shape: {outputs[1].shape}")
    print(f"output 3 shape: {outputs[2].shape}")
    print(f"output 4 shape: {outputs[3].shape}")
    flops, params = profile(double_nextv2, inputs=(dummy_rgb, dummy_depth))
    print(f"-------------------------DoubleNeXtV2-------------------------")
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))
    
    from models.encoders.convnextv2 import convnextv2_nano, convnextv2_atto
    nano = convnextv2_nano(drop_path_rate=0.1)
    flops, params = profile(nano, inputs=(dummy_rgb, ))
    print(f"-------------------------convnextv2_nano-------------------------")
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))
    atto = convnextv2_atto(drop_path_rate=0.1)
    flops, params = profile(atto, inputs=(dummy_rgb, ))
    print(f"-------------------------convnextv2_atto-------------------------")
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))

    from models.net_utils import FeatureFusionModule as FFM
    from models.net_utils import FeatureRectifyModule as FRM

    dims = [80, 160, 320, 640]  # only for rgb_backbone: convnextv2_N
    num_heads = [1, 2, 4, 8]
    norm_fuse = nn.BatchNorm2d
    
    FRMs = nn.ModuleList([FRM(dim=dims[i], reduction=1) for i in range(4)])
    FFMs = nn.ModuleList([FFM(dim=dims[i], 
                                       reduction=1,
                                       num_heads=num_heads[i],
                                       norm_layer=norm_fuse) for i in range(4)])
    flops_list = []
    params_list = []
    for i in range(4):
        flops, params = profile(FRMs[i], inputs=(outputs[i], outputs[i]))
        flops_list.append(flops)
        params_list.append(params)
        flops, params = profile(FFMs[i], inputs=(outputs[i], outputs[i]))
        flops_list.append(flops)
        params_list.append(params)
    flops = np.array(flops_list).sum()
    params = np.array(params_list).sum()
    print(f"-------------------------feature fusion-------------------------")
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))


def test_feature_fusion():
    print(f"-------------------------test_feature_fusion-------------------------")
    from models.net_utils import FeatureFusionModule as FFM
    from models.net_utils import FeatureRectifyModule as FRM

    dims = [64, 128, 320, 512]
    num_heads = [1, 2, 5, 8]
    H, W = 480, 640

    inputs = [torch.randn(1, dims[0], H//4, W//4),
              torch.randn(1, dims[1], H//8, W//8),
              torch.randn(1, dims[2], H//16, W//16),
              torch.randn(1, dims[3], H//32, W//32),]
    
    norm_fuse = nn.BatchNorm2d
    FRMs = nn.ModuleList([FRM(dim=dims[i], reduction=1) for i in range(4)])
    FFMs = nn.ModuleList([FFM(dim=dims[i], 
                                       reduction=1,
                                       num_heads=num_heads[i],
                                       norm_layer=norm_fuse) for i in range(4)])
    flops_list = []
    params_list = []
    for i in range(4):
        flops, params = profile(FRMs[i], inputs=(inputs[i], inputs[i]))
        flops_list.append(flops)
        params_list.append(params)
        flops, params = profile(FFMs[i], inputs=(inputs[i], inputs[i]))
        flops_list.append(flops)
        params_list.append(params)
    flops = np.array(flops_list).sum()
    params = np.array(params_list).sum()
    print(f"-------------------------feature fusion-------------------------")
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))


def test_DNeXtV2():
    from models.encoders.Dnext_v2 import DNeXtV2

    dummy_rgb, dummy_depth = torch.randn(1, 3, 512, 1024), torch.randn(1, 3, 512, 1024)
    dnext = DNeXtV2(rgb_backbone="N", 
                    d_backbone="A", 
                    downsample_ratio=1.0, 
                    output_to_depth=True, 
                    stage1_scc=False,
                    drop_path_rate=0.1)
    dnext.init_weights(pretrained_rgb=None,
                       pretrained_d=None)
    outputs = dnext(dummy_rgb, dummy_depth)
    print(f"output 1 shape: {outputs[0].shape}")
    print(f"output 2 shape: {outputs[1].shape}")
    print(f"output 3 shape: {outputs[2].shape}")
    print(f"output 4 shape: {outputs[3].shape}")
    flops, params = profile(model=dnext, inputs=(dummy_rgb, dummy_depth))
    print(f"-------------------------DNeXtV2-------------------------")
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))


if __name__ == "__main__":
    # test_asym_rgbd()
    # learn_layernorm()
    # test_double_NeXtV2()
    # test_feature_fusion()
    test_DNeXtV2()

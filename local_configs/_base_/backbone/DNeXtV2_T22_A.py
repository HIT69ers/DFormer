from .. import *

C.backbone = "DNeXtV2"
C.pretrained_model = ""
C.rgb_branch = "T"
C.rgb_pretrained = "/mnt/syh/pretrained/ConvNeXtV2/ImageNet-22K/convnextv2_tiny_22k_384_ema.pt"
C.d_branch = "A"  # Remember change the path below.
C.d_pretrained = "/mnt/syh/pretrained/ConvNeXtV2/ImageNet-1K/convnextv2_atto_1k_224_ema.pt"
C.downsample_ratio = 1.0  # Remember change the name below.
C.downsample_name = "full"
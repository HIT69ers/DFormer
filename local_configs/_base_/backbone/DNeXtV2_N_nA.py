from .. import *

C.backbone = "DNeXtV2"
C.pretrained_model = ""
C.rgb_branch = "N"
C.rgb_pretrained = "/mnt/syh/pretrained/ConvNeXtV2/ImageNet-22K/convnextv2_nano_22k_384_ema.pt"
C.d_branch = "A"  # Remember change the path below.
C.d_pretrained = None
C.downsample_ratio = 0.8  # Remember change the name below.
C.downsample_name = "eight"
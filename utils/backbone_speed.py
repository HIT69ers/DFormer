import os
import time
import random
import argparse
from importlib import import_module
import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from models.builder import EncoderDecoder as segmodel
import torch.nn as nn
from thop import profile

from models.encoders.mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from models.encoders.convnextv2 import *
from models.encoders.mscan import mscan_tiny, mscan_small, mscan_base, mscan_large
from models.encoders.mobilenet_v2 import mobilenetv2
from models.encoders.convnext import *


def get_backbone():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", '-b', help="train config file path")
    args = parser.parse_args()
    print(f"==========Testing Speed of {args.backbone}==========")
    backbones = dict(
        mit_b0=mit_b0,
        mit_b1=mit_b1,
        mit_b2=mit_b2,
        mit_b3=mit_b3,
        mit_b4=mit_b4,
        mit_b5=mit_b5,
        convnextv2_atto=convnextv2_atto,
        convnextv2_femto=convnextv2_femto,
        convnext_pico=convnext_pico,
        convnextv2_nano=convnextv2_nano,
        convnextv2_tiny=convnextv2_tiny,
        convnextv2_base=convnextv2_base,
        convnextv2_large=convnextv2_large,
        convnextv2_huge=convnextv2_huge,
        mscan_tiny=mscan_tiny,
        mscan_small=mscan_small,
        mscan_base=mscan_base,
        mscan_large=mscan_large,
        mobilenetv2=mobilenetv2,
        convnext_tiny=convnext_tiny,
        convnext_small=convnext_small,
        convnext_base=convnext_base,
        convnext_large=convnext_large,
        convnext_xlarge=convnext_xlarge
    )
    if args.backbone not in backbones.keys():
        raise NotImplementedError
    return backbones[args.backbone]()


if __name__ == '__main__':

    device = torch.device('cuda')
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.benchmark = True

    model = get_backbone()
    model.eval()
    model.to(device)

    dummy_input = torch.rand(1, 3, 480, 640).cuda()
    # dummy_input = torch.rand(1, 3, 240, 320).cuda()
    print(f"dummy_input.shape: {dummy_input.shape}")

    iterations = None

    str_time = time.strftime(f"%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Current time: {str_time}")

    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(dummy_input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(dummy_input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(round(FPS, 2))
    print("================================")
    flops, params = profile(model, inputs=(dummy_input, ))
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))
    print(f"-------------------------")

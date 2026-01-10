import argparse
from models.builder import EncoderDecoder as segmodel
import numpy as np
import torch
import torch.nn as nn
from importlib import import_module
from thop import profile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--iter_nums", help="number of iteration when getting mean inference speed", type=int, default=300)
    args = parser.parse_args()
    # config network and criterion
    config = getattr(import_module(args.config), "C")
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    device = torch.device("cuda:0")
    model.eval()
    model.to(device)
    dump_input = torch.ones(1, 3, 480, 640).to(device)
    input_shape = (3, 480, 640)
    flops, params = profile(model, inputs=(dump_input, dump_input))
    print("the flops is {}G,the params is {}M".format(round(flops / (10**9), 2), round(params / (10**6), 2)))

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Warm up
    for _ in range(50):
        _ = model(dump_input, dump_input)

    # Eval inference speed
    times = torch.zeros(int(args.iter_nums))
    with torch.no_grad():
        for iter in range(args.iter_nums):
            starter.record()
            _ = model(dump_input, dump_input)
            ender.record()
            torch.cuda.synchronize()
            current_time = starter.elapsed_time(ender)
            times[iter] = current_time

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))


PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    python utils/backbone_speed.py \
    -b="convnext_tiny"
    
# mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
# convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
# convnextv2_atto, convnextv2_femto, convnext_pico, convnextv2_nano, convnextv2_tiny, convnextv2_base, convnextv2_large, convnextv2_huge
# mscan_tiny, mscan_small, mscan_base, mscan_large
# mobilenetv2

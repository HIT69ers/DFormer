export CUDA_VISIBLE_DEVICES="1"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    python utils/latency.py \
    --config=local_configs.NYUDepthv2.DoubleMiT_b2_b0_nine \
    --pth_path="/mnt/syh/checkpoints/NYUDepthv2_DoubleMiT_mit_b2_mit_b0_nine_20251121-105851/epoch-478_miou_53.29.pth"
    
    
# config for DoubleMiTs on NYUDepthv2
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_five
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_six
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_seven
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_eight
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_nine
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_full
# local_configs.NYUDepthv2.DoubleMiT_b1_b0_five
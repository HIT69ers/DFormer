export CUDA_VISIBLE_DEVICES="1"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    python utils/DDRNet_23_slim_eval_speed.py \
    --config=local_configs.NYUDepthv2.DoubleMiT_b2_b0_seven 
    
    
# config for DoubleMiTs on NYUDepthv2
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_five
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_six
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_seven
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_eight
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_nine
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_full
# local_configs.NYUDepthv2.DoubleMiT_b1_b0_five

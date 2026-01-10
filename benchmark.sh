export CUDA_VISIBLE_DEVICES="5"

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    python utils/benchmark.py \
    --config=local_configs.NYUDepthv2.DoubleMiT_b0_b0_five \
    --iter_nums=300

# config for DoubleMiTs on NYUDepthv2
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_five
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_six
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_seven
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_eight
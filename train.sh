GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29158}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="0,1"
export TORCHDYNAMO_VERBOSE=1

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.NYUDepthv2.DoubleMiT_b0_b0_nine --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --no-mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --pad_SUNRGBD \
    --use_seed \
    -c="/mnt/syh/checkpoints/NYUDepthv2_DoubleMiT_mit_b0_mit_b0_nine_20251128-221239/epoch-364_miou_43.36.pth"

# config for DFormers on NYUDepthv2
# local_configs.NYUDepthv2.DFormer_Large
# local_configs.NYUDepthv2.DFormer_Base
# local_configs.NYUDepthv2.DFormer_Small
# local_configs.NYUDepthv2.DFormer_Tiny
# local_configs.NYUDepthv2.DFormer_v2_S
# local_configs.NYUDepthv2.DFormer_v2_B
# local_configs.NYUDepthv2.DFormer_v2_L

# config for DFormers on SUNRGBD
# local_configs.SUNRGBD.DFormer_Large
# local_configs.SUNRGBD.DFormer_Base
# local_configs.SUNRGBD.DFormer_Small
# local_configs.SUNRGBD.DFormer_Tiny
# local_configs.SUNRGBD.DFormer_v2_S
# local_configs.SUNRGBD.DFormer_v2_B
# local_configs.SUNRGBD.DFormer_v2_L

# config for DoubleMiTs on NYUDepthv2
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_five
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_six
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_seven
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_eight
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_nine
# local_configs.NYUDepthv2.DoubleMiT_b0_b0_full
# local_configs.NYUDepthv2.DoubleMiT_b1_b0_five

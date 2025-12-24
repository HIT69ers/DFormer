GPUS=2
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29159}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="2,3"
export TORCHDYNAMO_VERBOSE=1

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=local_configs.Cityscapes.DNeXtV2.N_A_False_five_LMLP256_6_500_1_b12 --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --no-mst \
    --compile_mode="default" \
    --no-amp \
    --val_amp \
    --pad_SUNRGBD \
    --use_seed 

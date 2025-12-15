from .._base_.datasets.NYUDepthv2 import *

""" Settings for network, this would be different for each kind of model"""
# C.backbone = "DFormer-Base"  # Remember change the path below.
# C.pretrained_model = "checkpoints/pretrained/DFormer_Base.pth.tar"
# C.decoder = "ham"
# C.decoder_embed_dim = 512
# C.optimizer = "AdamW"
C.backbone = "DoubleMiT"
C.pretrained_model = ""
C.rgb_branch = "mit_b0"  # Remember change the path below.
C.rgb_pretrained = "/mnt/syh/pretrained/segformer/mit_b0.pth"
C.d_branch = "mit_b0"  # Remember change the path below.
C.d_pretrained = "/mnt/syh/pretrained/segformer/mit_b0.pth"
C.downsample_ratio = 0.8  # Remember change the name below.
C.downsample_name = "eight"
C.decoder = "MLPDecoder"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 500
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 16
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate = 0.1
C.aux_rate = 0

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # [0.75, 1, 1.25] #
C.eval_flip = True  # False #
C.eval_crop_size = [480, 640]  # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 250
C.checkpoint_step = 25

"""Path Config"""
C.name = C.backbone + '_' + C.rgb_branch + '_' + C.d_branch + '_' + C.downsample_name
C.log_dir = osp.abspath("/mnt/syh/checkpoints/" + C.dataset_name + "_" + C.name)
C.log_dir = C.log_dir + "_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()).replace(" ", "_")
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(
    osp.join(C.log_dir, "checkpoint")
)  #'/mnt/sda/repos/2023_RGBX/pretrained/'#osp.abspath(osp.join(C.log_dir, "checkpoint"))
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir, exist_ok=True)
exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

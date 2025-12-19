from ..._base_.backbone.DNeXtV2_N_A import *
from ..._base_.decoder.LMLP256 import *
from ..._base_.datasets.Cityscapes import *
from ..._base_.schedule.p6_1000_1 import *

C.downsample_ratio = 1.0  # Remember change the name below.
C.downsample_name = "full"

C.output_to_depth = False  # If adding fused feature maps into depth backbone
C.stage1_scc = False  # If true, adding feature fusion after stage 1

"""Path Config"""
C.name = C.backbone + '_' + C.rgb_branch + '_' + C.d_branch + '_' + C.downsample_name
if C.d_pretrained is None:
    C.d_branch = "n" + C.d_branch
if C.output_to_depth:
    C.name = "T_" + C.name
else:
    C.name = "F_" + C.name

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
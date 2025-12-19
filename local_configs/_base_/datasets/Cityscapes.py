from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "Cityscapes"
C.dataset_path = osp.join(C.root_dir, "Cityscapes")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".png"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = False
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
C.x_root_folder = osp.join(C.dataset_path, "HHA")
C.x_format = ".png"
C.x_modal = ["hha"]
C.x_is_single_channel = None
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "val.txt")
C.num_train_imgs = 2975
C.num_eval_imgs = 500
C.num_classes = 19
C.class_names = [
    'road', 
    'sidewalk', 
    'building', 
    'wall', 
    'fence', 
    'pole',
    'traffic light', 
    'traffic sign', 
    'vegetation', 
    'terrain',
    'sky', 
    'person', 
    'rider', 
    'car', 
    'truck', 
    'bus', 
    'train',
    'motorcycle', 
    'bicycle'
]

"""Image Config"""
C.background = 255
C.image_height = 512
C.image_width = 1024
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])
import os.path as osp

from configs.data_config import DataConfig as data_cfg
from utils.utils import split_to_train_test

if __name__ == '__main__':
    split_to_train_test(osp.join(data_cfg.detection_out_path, 'VOC2007'))

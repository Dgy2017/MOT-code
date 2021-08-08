import sys

sys.path.append('../')
import numpy as np
import cv2
from configs.data_config import DataConfig as data_cfg
from configs.alignedreid_config import AlignedReIdConfig as reid_cfg
from dataloader.dataloader import DukeMTMCTVideo
from models.resnet import resnet50_v1
from models.km import kuhn_munkres
import os
from mxnet import nd
from utils.utils import compute_overlaps
from utils import utils
from configs.pipeline_config import PipelineConfig as pl_cfg
from models.loss import euler_distance
from models.entity import Trajectory, FrameInfo, Galley
import time
import gluoncv as gcv
import motmetrics as mm
from gluoncv.data.transforms import presets

try:
    with open('end_frame.txt') as f:
        pl_cfg.end_frame = int(f.readlines()[0])
except:
    print('open failed')

methods = ['original','presearch','weighted','weighted_presearch']
for method in methods:
    save_dict = utils.load_pickle(f'/data/stu06/homedir/project/gluon/MCMT/pipeline/evaluator_{method}.pkl')
    acc = save_dict['acc']
    mh = save_dict['mh']
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr','mota','motp'], name=method)
    print(str(summary))
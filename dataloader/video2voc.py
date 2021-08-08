#!/usr/bin/env python
# 参考来源：https://github.com/veraposeidon/labelme2Datasets
import sys

sys.path.append('../')
from tqdm import tqdm
from configs.data_config import DataConfig as data_cfg
from configs.pipeline_config import PipelineConfig as pl_cfg
from dataloader import DukeMTMCTVideo
from utils.utils import voc_format, make_sure_dir
import numpy as np
import os
import os.path as osp


def main(video_stream):
    # 创建PASCAL格式数据集文件夹
    det_path = os.path.join(data_cfg.detection_out_path, 'VOC2007')
    make_sure_dir(det_path)
    make_sure_dir(osp.join(det_path, 'JPEGImages'))
    make_sure_dir(osp.join(det_path, 'Annotations'))
    # 整理类型名称,就保存成英文吧
    class_names = data_cfg.detection_classes

    object_num = 8543
    last_frame_pos = 0
    try:
        for frame_pos, frames, bboxes, gt_bboxes in tqdm(video_stream):
            # for frame,gt_bbox in zip(frames,gt_bboxes):
            last_frame_pos = frame_pos[-1]
            frame, gt_bbox = frames[0], gt_bboxes[0]
            if gt_bbox is not None and len(gt_bbox) > 0:
                gt_bbox = gt_bbox.astype(np.int32)
                gt_bbox[:, 2:4] = gt_bbox[:, 2:4] + gt_bbox[:, 0:2]
                voc_format(frame, '%06d' % object_num, gt_bbox,
                           det_path,
                           [class_names[0]] * len(gt_bbox))
                object_num += 1
    except Exception as e:
        print(e.__traceback__)
        print(f'last frame position is {last_frame_pos}')
        print(f'last object number is {object_num}')


if __name__ == '__main__':
    start_frame_pos = pl_cfg.cam1_gt_start_frame_pos
    cur_frame_pos = start_frame_pos
    video_stream = DukeMTMCTVideo(video_path=data_cfg.video_path,
                                  camera_id=4,
                                  bbox_path=data_cfg.bbox_path,
                                  gt_bbox_path=os.path.join(data_cfg.gt_path),
                                  batch_size=60,
                                  start_frame_pos=start_frame_pos)

    main(video_stream)

import sys

sys.path.append('../')
import numpy as np
from dataloader.dataloader import DukeMTMCTVideo
from utils.utils import dump_pickle, load_pickle
import os
from configs.pipeline_config import PipelineConfig as pl_cfg
import gluoncv as gcv
from gluoncv.data.transforms import presets

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import time
from mxnet import nd
from gluoncv.model_zoo import get_model
from configs.data_config import DataConfig as data_cfg


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def yolo_detector(detector_name='yolo3_darknet53_coco', video_stream=None):
    assert video_stream is not None

    start_time = time.time()
    detector = gcv.model_zoo.get_model(detector_name, pretrained=pl_cfg.detector_pretrained,
                                       root=pl_cfg.detector_path)

    if not pl_cfg.detector_pretrained:
        # todo 修改加载路径
        detector.load_parameters('...')
    detector.collect_params().reset_ctx(pl_cfg.detector_ctx)
    # detector.hybridize()
    detector.set_nms(pl_cfg.detector_nms_thresh, pl_cfg.detector_nms_topk)
    cur_frame_pos = 0
    bbox_dict = dict()
    for frame_pos, frames in video_stream:

        frames = [nd.array(img) for img in frames]
        x, imgs = presets.yolo.transform_test(frames, short=512)
        x = nd.concatenate(x, axis=0)
        x = x.as_in_context(pl_cfg.detector_ctx)
        ids, scores, bboxes = detector(x)
        # detector.export('yolo')
        ids, scores, bboxes = ids.asnumpy(), scores.asnumpy(), bboxes.asnumpy()
        for frame, id, score, bbox in zip(list(imgs), list(ids), list(scores), list(bboxes)):
            id = np.squeeze(id, axis=1)

            bbox_filter, = np.nonzero(id == 0)
            score = np.squeeze(score[bbox_filter], axis=1)

            bbox = bbox[bbox_filter]
            bbox = bbox[np.nonzero(score > 0.5)[0]]
            scale_height, scale_width = frame.shape[0], frame.shape[1]
            bbox[:, (0, 2)] *= video_stream.w / scale_width
            bbox[:, (1, 3)] *= video_stream.h / scale_height
            bbox = bbox.astype(np.int32)

            cur_frame_pos += 1
            bbox_dict[cur_frame_pos] = bbox if len(bbox) > 0 else None

        delta_time = time.time() - start_time
        print(f"{cur_frame_pos} frame was processed."
              f" Speed:{delta_time/cur_frame_pos:.4f}s/f Elapse:{delta_time/3600:.2f}h")
        if cur_frame_pos % 2000 == 0:
            dump_pickle(bbox_dict, pl_cfg.yolo_bbox_path)


def faster_rcnn_detector(detector_name='faster_rcnn_resnet50_v1b_voc', video_stream=None, break_point=10000):
    classes = data_cfg.detection_classes
    # network

    detector = get_model(detector_name, pretrained=False)
    detector.reset_class(classes)
    if pl_cfg.detector_pretrained:
        # todo 修改加载路径
        detector.load_parameters(pl_cfg.local_detector)
    detector.collect_params().reset_ctx(pl_cfg.detector_ctx)
    detector.set_nms(pl_cfg.detector_nms_thresh, pl_cfg.detector_nms_topk)

    bbox_dict = dict() if break_point == 1 else load_pickle(pl_cfg.faster_rcnn_bbox_path)
    start_time = time.time()
    cur_frame_pos = break_point
    for frame_pos, frames in video_stream:

        frames = [nd.array(img) for img in frames]
        x, imgs = presets.rcnn.transform_test(frames, short=512)
        if not isinstance(imgs, list) and len(imgs.shape) == 3:
            imgs = [imgs]
        # x = nd.concatenate(x, axis=0)
        if isinstance(pl_cfg.detector_ctx, list):
            x = split_and_load([x], pl_cfg.detector_ctx)
            ids, scores, bboxes = [], [], []
            for xi in x[0]:
                t1, t2, t3 = detector(xi)
                ids.append(t1)
                scores.append(t2)
                bboxes.append(t3)
            # ids,scores,bboxes = nd.concatenate(ids,axis=0),nd.concatenate(scores,axis=0),nd.concatenate(bboxes,axis=0)
        else:
            x = x.as_in_context(pl_cfg.detector_ctx)
            ids, scores, bboxes = detector(x)
        # ids, scores, bboxes = ids.asnumpy(), scores.asnumpy(), bboxes.asnumpy()
        for cur_frame_pos, frame, id, score, bbox in zip(frame_pos, imgs, ids, scores, bboxes):
            id = id.asnumpy().squeeze()
            score = score.asnumpy().squeeze()
            bbox = bbox.asnumpy().squeeze()

            bbox_filter, = np.nonzero(id == 0)
            score = score[bbox_filter]

            bbox = bbox[bbox_filter]
            bbox = bbox[np.nonzero(score > 0.5)[0]]
            scale_height, scale_width = frame.shape[0], frame.shape[1]
            bbox[:, (0, 2)] *= video_stream.w / scale_width
            bbox[:, (1, 3)] *= video_stream.h / scale_height
            bbox = bbox.astype(np.int32)
            bbox = np.clip(bbox, a_min=0, a_max=1920)
            bbox_dict[cur_frame_pos] = bbox if len(bbox) > 0 else None

        delta_time = time.time() - start_time
        if cur_frame_pos % 40 == 0:
            print(f"{cur_frame_pos} frame was processed."
                  f" Speed:{delta_time/cur_frame_pos:.4f}s/f Elapse:{delta_time/3600:.2f}h")
        if cur_frame_pos % 2000 == 0:
            dump_pickle(bbox_dict, pl_cfg.faster_rcnn_bbox_path)


def ssd_detector(detector_name='ssd_300_vgg16_atrous_coco', video_stream=None):
    assert video_stream is not None

    start_time = time.time()
    detector = gcv.model_zoo.get_model(detector_name, pretrained=pl_cfg.detector_pretrained,
                                       root=pl_cfg.detector_path)
    print(detector.classes)
    if not pl_cfg.detector_pretrained:
        # todo 修改加载路径
        detector.load_parameters('...')
    detector.collect_params().reset_ctx(pl_cfg.detector_ctx)
    detector.set_nms(pl_cfg.detector_nms_thresh, pl_cfg.detector_nms_topk)
    cur_frame_pos = 0
    bbox_dict = dict()
    for _, frames, _ in video_stream:

        frames = [nd.array(img) for img in frames]
        x, imgs = presets.ssd.transform_test(frames, short=512)
        x = nd.concatenate(x, axis=0)
        x = x.as_in_context(pl_cfg.detector_ctx)
        ids, scores, bboxes = detector(x)
        ids, scores, bboxes = ids.asnumpy(), scores.asnumpy(), bboxes.asnumpy()
        for frame, id, score, bbox in zip(list(imgs), list(ids), list(scores), list(bboxes)):
            id = np.squeeze(id, axis=1)

            bbox_filter, = np.nonzero(id == 0)
            score = np.squeeze(score[bbox_filter], axis=1)

            bbox = bbox[bbox_filter]
            bbox = bbox[np.nonzero(score > 0.5)[0]]
            scale_height, scale_width = frame.shape[0], frame.shape[1]
            bbox[:, (0, 2)] *= video_stream.w / scale_width
            bbox[:, (1, 3)] *= video_stream.h / scale_height
            bbox = bbox.astype(np.int32)

            cur_frame_pos += 1
            bbox_dict[cur_frame_pos] = bbox if len(bbox) > 0 else None

        delta_time = time.time() - start_time
        print(f"{cur_frame_pos} frame was processed."
              f" Speed:{delta_time/cur_frame_pos:.4f}s/f Elapse:{delta_time/3600:.2f}h")
        if cur_frame_pos % 2000 == 0:
            dump_pickle(bbox_dict, pl_cfg.ssd_bbox_path)


if __name__ == '__main__':
    camera_id = 1
    video_stream = DukeMTMCTVideo(video_path=data_cfg.video_path, camera_id=camera_id,
                                  batch_size=2,
                                  start_frame_pos=1)
    # faster_rcnn_detector(video_stream=video_stream, break_point=1)

    yolo_detector(video_stream=video_stream)
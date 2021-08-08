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

method_name = 'weighted'

def normalize(imgs, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):
    if isinstance(imgs, list):
        imgs = np.array(imgs)
    if len(imgs.shape) == 4:
        img_mean = np.array(img_mean).reshape((1, 1, 1, 3))
        img_std = np.array(img_std).reshape((1, 1, 1, 3))
    else:
        img_mean = np.array(img_mean).reshape((1, 1, 3))
        img_std = np.array(img_std).reshape((1, 1, 3))
    imgs = (imgs / 225.0 - img_mean) / img_std
    return imgs


def get_people(img, bboxes,nl=True, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):
    # img : RGB
    imgs = []

    bboxes = bboxes.astype(np.int32)  # todo
    for bbox in np.vsplit(bboxes, indices_or_sections=bboxes.shape[0]):
        bbox = bbox[0]
        if bbox[3] - bbox[1] > 1 and bbox[2] - bbox[0] > 1:
            people = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # img = cv2.rectangle(img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color=123456)
            imgs.append(cv2.resize(people, dsize=(reid_cfg.input_width, reid_cfg.input_height)))
            # cv2.imwrite('temp_frame.jpg', cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR))
    if nl:
        return normalize(imgs)
    else:
        return imgs


def get_tra_frame_weight(bbox1, tras:list, bbox2, feature2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = bbox1.asnumpy()
    if not isinstance(bbox2, np.ndarray):
        bbox2 = bbox2.asnumpy()
    iou_matrix = compute_overlaps(bbox1, bbox2)
    dist_matrix = []
    for tra in tras:
        dist_matrix.append(tra.feat_distance(feature2))
    dist_matrix = nd.stack(*dist_matrix,axis=0)
    dist_matrix = dist_matrix.asnumpy()

    graph = pl_cfg.s - dist_matrix - np.where(iou_matrix < pl_cfg.k, pl_cfg.inf, 0)
    return graph


def get_frame_frame_weight(bbox1, feature1, bbox2, feature2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = bbox1.asnumpy()
    if not isinstance(bbox2, np.ndarray):
        bbox2 = bbox2.asnumpy()
    iou_matrix = compute_overlaps(bbox1, bbox2)
    dist_matrix = euler_distance(feature1, feature2)
    dist_matrix = dist_matrix.asnumpy()

    graph = pl_cfg.s - dist_matrix - np.where(iou_matrix < pl_cfg.k, pl_cfg.inf, 0)
    return graph


def tra_bbox_match(trajectories, features, bboxes, cur_frame, old_frame_idx):
    weight = []
    assert len(trajectories) > 0
    for tra in trajectories:
        weight.append(tra.bbox_distance(old_frame_idx, features, bboxes, cur_frame))
    weight = np.row_stack(weight)
    km = kuhn_munkres(weight)
    match = km()
    assert len(match) == bboxes.shape[0]
    # todo train a more precise ReId model
    for i in range(len(match)):
        if match[i] >= 0:
            tra = trajectories[match[i]]
            feat_dist = tra.feat_distance(features[i])
            feat_dist = feat_dist.mean()
            l, r, sigma, u = tra.ideal_region(3, detail=True)
            if feat_dist > r+0.2:
                match[i] = -1
    return match


def process(gallery, frame: np.ndarray, bbox: np.ndarray, gt_bbox: np.ndarray, reid_model, cur_frame_pos):
    if bbox is not None and len(bbox) > 0:
        bbox = bbox.astype(np.int32)
        people = get_people(frame, bbox)
        feature = reid_model(nd.array(people, ctx=pl_cfg.reid_ctx).transpose(axes=(0, 3, 1, 2)))
        feature = feature[0]
        feature = feature / (nd.norm(feature, axis=1, keepdims=True, ord=2) + 1e-12)
        feature = feature.as_in_context(Trajectory.context)
        cur_frame_info = FrameInfo(np.array([None] * bbox.shape[0]), frame, bbox, feature, cur_frame_pos,
                                   gt_bbox=gt_bbox)
        pre_frame_info = gallery.top()
        # cur_color = get_new_color(len(feature))
        matched_bool_index = np.array([False] * bbox.shape[0])
        unmatched_bool_index = np.logical_not(matched_bool_index)
        if pre_frame_info is not None and pre_frame_info.bboxes is not None:
            # graph = get_frame_frame_weight(pre_frame_info.bboxes, pre_frame_info.features, bbox, feature)
            graph = get_tra_frame_weight(pre_frame_info.bboxes, [tra for tra in pre_frame_info.trajectory_array], bbox, feature)

            km = kuhn_munkres(graph)
            match = km()
            # pre_color = pre_frame[2]
            matched_bool_index = (match >= 0)
            unmatched_bool_index = np.logical_not(matched_bool_index)

            cur_frame_info.trajectory_array[matched_bool_index] = pre_frame_info.trajectory_array[
                match[matched_bool_index]]
        # if any(unmatched_bool_index):
        #     # There are some bboxes did not match trajectories
        #     matched_tra = set(cur_frame_info.trajectory_array[matched_bool_index])
        #     for old_frame_info in gallery.reverse_slice(pl_cfg.gallery_step):
        #         if old_frame_info.trajectory_array is None:
        #             continue
        #         unmatched_tra = np.array(list(set(old_frame_info.trajectory_array) - matched_tra))
        #         nd_indices, = unmatched_bool_index.nonzero()
        #         nd_indices = nd.array(nd_indices)
        #         if len(unmatched_tra) <= 0:
        #             continue
        #         m = tra_bbox_match(unmatched_tra, feature.take(nd_indices),
        #                            bbox[unmatched_bool_index], cur_frame_pos, old_frame_info.frame_idx)
        #         m_bool_idx = (m > -1)
        #         if any(m_bool_idx):
        #             tra_array = np.array([None] * m_bool_idx.shape[0])
        #             tra_array[m_bool_idx] = unmatched_tra[m[m_bool_idx]]
        #             # todo 修改算法
        #
        #             cur_frame_info.trajectory_array[unmatched_bool_index] = tra_array
        #             matched_bool_index[unmatched_bool_index] = m_bool_idx
        #             unmatched_bool_index = np.logical_not(matched_bool_index)
        #
        #         if all(matched_bool_index):
        #             break
        for i in range(bbox.shape[0]):
            if cur_frame_info.trajectory_array[i] is None:
                cur_frame_info.trajectory_array[i] = Trajectory(feature[i], bbox[i], cur_frame_pos)
            else:
                dist = cur_frame_info.trajectory_array[i].feat_distance(people=feature[i])
                cur_frame_info.trajectory_array[i].add_people(feature[i], bbox[i], cur_frame_pos)
                cur_frame_info.trajectory_array[i].add_dist(dist)
    else:
        cur_frame_info = FrameInfo(None, frame, None, None, frame_idx=cur_frame_pos, gt_bbox=gt_bbox)
    return cur_frame_info


def pipeline(camera_id=5, start_frame_pos=1):
    # start_frame_pos = pl_cfg.cam1_gt_start_frame_pos

    video_stream = DukeMTMCTVideo(video_path=data_cfg.video_path,
                                  bbox_path=pl_cfg.faster_rcnn_bbox_path,
                                  camera_id=camera_id,
                                  gt_bbox_path=data_cfg.gt_path,
                                  batch_size=40,
                                  start_frame_pos=start_frame_pos)
    reid_model = resnet50_v1(ctx=pl_cfg.reid_ctx, use_fc=reid_cfg.use_fc, classes=reid_cfg.classes)
    reid_model.load_parameters(os.path.join(reid_cfg.check_point, 'model_0.878_margin_0.3.params'),
                               ctx=pl_cfg.reid_ctx)
    reid_model.collect_params().reset_ctx(ctx=pl_cfg.reid_ctx)


    acc = mm.MOTAccumulator(auto_id=True)
    mh = mm.metrics.create()

    gallery = Galley(f'fragment_{method_name}_', pl_cfg.prior_search_length, cv2.COLOR_RGB2BGR, acc,
                     video_stream.fps, video_stream.w, video_stream.h,version=1)
    start_time = time.time()
    print(f'Video start at {start_frame_pos}th frame')
    if pl_cfg.use_detector:
        detector = gcv.model_zoo.get_model(pl_cfg.detector_name, pretrained=pl_cfg.detector_pretrained,
                                           root=pl_cfg.detector_path)

        if not pl_cfg.detector_pretrained:
            # todo 修改加载路径
            detector.load_parameters(pl_cfg.local_detector)
        detector.collect_params().reset_ctx(pl_cfg.detector_ctx)
        detector.set_nms(pl_cfg.detector_nms_thresh, pl_cfg.detector_nms_topk)
        for frame_pos_array, frames in video_stream:
            imgs = frames.copy()
            frames = [nd.array(img) for img in frames]
            x, _ = presets.yolo.transform_test(frames, short=512)
            x = nd.concatenate(x, axis=0)
            x = x.as_in_context(pl_cfg.detector_ctx)
            ids, scores, bboxes = detector(x)
            ids, scores, bboxes = ids.asnumpy(), scores.asnumpy(), bboxes.asnumpy()
            for cur_frame_pos, frame, id, score, bbox in zip(frame_pos_array, list(imgs), list(ids), list(scores),
                                                             list(bboxes)):
                id = np.squeeze(id, axis=1)

                bbox_filter, = np.nonzero(id == 0)
                score = np.squeeze(score[bbox_filter], axis=1)

                bbox = bbox[bbox_filter]
                bbox = bbox[np.nonzero(score > 0.5)[0]]

                scale_height, scale_width = frame.shape[0], frame.shape[1]
                if scale_width != video_stream.w:
                    bbox[:, (0, 2)] *= video_stream.w / scale_width
                    bbox[:, (1, 3)] *= video_stream.h / scale_height

                cur_frame_info = process(gallery, frame, bbox, reid_model, cur_frame_pos)
                gallery.add_frame_info(cur_frame_info)

            delta_time = time.time() - start_time
            print(
                f"{cur_frame_pos} frame was processed. Speed:{delta_time/cur_frame_pos:.4f}s/f Elapse:"
                f"{delta_time/3600:.2f}h")
            if cur_frame_pos > 10000:
                break
    else:
        iter_num = 0
        for frame_pos_array, frames, bboxes, gt_bboxes in video_stream:
            # frames:np.ndarray,RGB,(H,W,3)，PIL format
            for cur_frame_pos, frame, bbox, gt_bbox in zip(frame_pos_array, frames, bboxes, gt_bboxes):
                gt_bbox = gt_bbox.astype(np.int32) if gt_bbox is not None else None

                cur_frame_info = process(gallery, frame, bbox, gt_bbox, reid_model, cur_frame_pos)
                gallery.add_frame_info(cur_frame_info)
            # try:
            #     summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr'], name='acc')
            #     print('\033[32m' + str(summary))
            # except:
            #     pass
            if iter_num % 20 == 0:
                save_dict = dict(
                    acc=acc,
                    mh=mh,
                    last_frame=frame_pos_array[-1],
                )
                utils.dump_pickle(save_dict,f'evaluator_{method_name}.pkl')
            delta_time = time.time() - start_time
            iter_num += 1
            # todo:delete
            if frame_pos_array[-1] > pl_cfg.end_frame:
                break

            print(
                f"\033[34m{cur_frame_pos} frame was processed. Speed:{delta_time/(cur_frame_pos-start_frame_pos):.4f}s/f "
                f"Elapse:{delta_time/3600:.2f}h")
    gallery.clear_gallery()



from utils.utils import make_sure_dir
def save_frame(camera_id=1, start_frame_pos=1,save_dir='./'):
    make_sure_dir(save_dir)
    video_stream = DukeMTMCTVideo(video_path=data_cfg.video_path,
                                  bbox_path=pl_cfg.faster_rcnn_bbox_path,
                                  camera_id=camera_id,
                                  gt_bbox_path=data_cfg.gt_path,
                                  batch_size=40,
                                  start_frame_pos=start_frame_pos)

    for frame_pos_array, frames, bboxes, gt_bboxes in video_stream:
        # frames:np.ndarray,RGB,(H,W,3)，PIL format
        for cur_frame_pos, frame, bbox, gt_bbox in zip(frame_pos_array, frames, bboxes, gt_bboxes):
            # todo 注释掉
            if bbox is not None:
                bbox = bbox.astype(np.int32)
                # frame = frame.astype(np.int)
                people = get_people(frame, bbox,nl=False)
                for i, p in enumerate(people):
                    p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_dir, f'{cur_frame_pos}_{i}.jpg'), p)
                for bb in bbox:
                    frame = cv2.rectangle(frame,tuple(bb[:2]),tuple(bb[2:4]),(0,255,0),thickness=3)
                cv2.imwrite(os.path.join(save_dir,f'{cur_frame_pos}_frame.jpg'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        break




if __name__ == '__main__':
    pipeline(camera_id=1,start_frame_pos=pl_cfg.gt_start_frame_pos[0])  #
    # pipeline(camera_id=1,start_frame_pos=1)
    # get_new_color()
    # save_frame(1,4340,'/data/stu06/MCMTDataSet/code/frame_snapshoot')

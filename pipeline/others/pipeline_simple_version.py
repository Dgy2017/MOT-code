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
from utils.utils import compute_overlaps,get_new_color
from configs.pipeline_config import PipelineConfig as pl_cfg
from models.loss import euler_distance
def get_people(img,bboxes):
    imgs = []
    bboxes = bboxes.astype(np.int32)# todo
    for bbox in np.vsplit(bboxes,indices_or_sections=bboxes.shape[0]):
        bbox = bbox[0]
        people = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        # img = cv2.rectangle(img,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color=123456)
        imgs.append(cv2.resize(people,dsize=(reid_cfg.input_width,reid_cfg.input_height)))
    # cv2.imwrite('temp_frame.jpg',img)
    return np.array(imgs)


def get_weight(bbox1,feature1,bbox2,feature2):
    if not isinstance(bbox1,np.ndarray):
        bbox1 = bbox1.asnumpy()
    if not isinstance(bbox2,np.ndarray):
        bbox2 = bbox2.asnumpy()
    iou_matrix = compute_overlaps(bbox1,bbox2)
    dist_matrix = euler_distance(feature1,feature2)
    dist_matrix = dist_matrix.asnumpy()

    graph = pl_cfg.s - dist_matrix - np.where(iou_matrix<pl_cfg.k,pl_cfg.inf,0)
    return graph


def pipeline(use_detector=False):
    video_stream = DukeMTMCTVideo(video_path=data_cfg.video_path,
                                  bbox_path=pl_cfg.other_bbox_path,
                                  camera_id=1,
                                  gt_bbox_path=data_cfg.gt_path,
                                  batch_size=40,
                                  start_frame_pos=1)
    reid_model = resnet50_v1(ctx=pl_cfg.reid_ctx, use_fc=reid_cfg.use_fc, classes=reid_cfg.classes)
    reid_model.load_parameters(os.path.join(reid_cfg.check_point, 'model_0.878.params'),
                               ctx=reid_cfg.ctx)
    reid_model.collect_params().reset_ctx(ctx=pl_cfg.reid_ctx)
    match_list = []
    pre_frame = None
    cur_frame = 0
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('fragment.avi', fourcc, video_stream.fps,
                                   (video_stream.w, video_stream.h))

    if use_detector:
        pass # todo use faster-rcnn detect people
    else:
        for frame_pos_array, frames, bboxes, gt_bboxes in video_stream:
            # frames:np.ndarray,RGB,(H,W,3)，PIL format
            for cur_frame_pos, frame, bbox, gt_bbox in zip(frame_pos_array, frames, bboxes, gt_bboxes):
                gt_bbox = gt_bbox.astype(np.int32) if gt_bbox is not None else None
                if bbox is not None:
                    bbox = bbox.astype(np.int32)
                    people = get_people(frame,bbox)
                    feature,_,_ = reid_model(nd.array(people,ctx=reid_cfg.ctx).transpose(axes=(0, 3, 1, 2)))
                    feature = feature / (nd.norm(feature, axis=1, keepdims=True, ord=2) + 1e-12)
                    cur_color = get_new_color(len(feature))
                    if pre_frame is not None:
                        graph = get_weight(*pre_frame[:2],bbox,feature)
                        km = kuhn_munkres(graph)
                        match = km()
                        pre_color = pre_frame[2]
                        for i in range(len(match)):
                            if match[i] > -1:
                                try:
                                    cur_color[i] = pre_color[match[i]]
                                except:
                                    print(d)
                        match_list.append((cur_frame, bbox, match))
                    pre_frame = (bbox, feature,cur_color)
                    for b,c in zip(list(bbox),cur_color):
                        frame = cv2.rectangle(frame,(b[0], b[1]), (b[2], b[3]), color=[int(cc) for cc in c],thickness=3)
                    video_writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                else:
                    pre_frame = None
                cur_frame += 1


            print(cur_frame)
            # todo
            if frame_pos_array[0] > 5000:
                break
        video_writer.release()


if __name__ == '__main__':
    pipeline()#
    # get_new_color()
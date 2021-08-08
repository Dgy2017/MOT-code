import os

from mxnet import gpu


class PipelineConfig:
    root = '/data/stu06/MCMTDataSet/'
    s = 0.5
    k = 0.2
    inf: float = 1e9
    max_interval = 150
    threshold_alpha = 0.5062
    threshold_beta = 0.8
    prior_search_length = 40
    gallery_step = 5
    tra_length_low_bound = 15
    reid_ctx = gpu(3)
    use_detector = False

    gt_start_frame_pos = [44158, 46094, 26078, 18913, 50742, 27476, 30733, 2935]

    # config detector
    # detector_ctx = [gpu(0),gpu(1),gpu(2),gpu(3)]
    detector_ctx = gpu(2)
    detector_path = os.path.join(root, 'code', 'detector')
    detector_name = 'yolo3_darknet53_coco'
    detector_pretrained = True
    detector_nms_thresh = 0.45
    detector_nms_topk = 50
    local_detector = '/data/stu06/homedir/project/gluon/MCMT/experiments/faster_rcnn_resnet50_v1b_voc_0018_0.9084.params'
    other_bbox_path = os.path.join(root, 'DukeMTMC/bbox/det-camera1.pkl')

    yolo_bbox_path = os.path.join(root, 'DukeMTMC/bbox/det-yolo-camera1.pkl')
    ssd_bbox_path = os.path.join(root, 'DukeMTMC/bbox/det-ssd-camera1.pkl')
    faster_rcnn_bbox_path = os.path.join(root, 'DukeMTMC/bbox/faster-rcnn-camera1.pkl')

    end_frame = 100000

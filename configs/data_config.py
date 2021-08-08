import os


class DataConfig(object):
    root = '/data/stu06/MCMTDataSet'

    cuhk03_mat_path = os.path.join(root, 'reid', 'cuhk03', 'cuhk-03.mat')
    cuhk03_pkl_path = os.path.join(root, 'reid', 'cuhk03', 'cuhk_03.pkl')

    market1501_root = os.path.join(root, 'reid', 'DukeMTMC-reID')

    video_path = os.path.join(root, 'DukeMTMC/')
    bbox_path = os.path.join(root, 'DukeMTMC/bbox/det-camera1.pkl')

    gt_path = os.path.join(root, 'DukeMTMC/videos/ground-truth')

    detection_out_path = os.path.join(root, 'DukeMTMC/detection/')
    detection_classes = ['person']

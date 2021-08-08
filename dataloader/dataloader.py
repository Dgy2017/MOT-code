from __future__ import absolute_import
from __future__ import division

import sys

sys.path.append('../')

from configs.data_config import DataConfig

from mxnet import nd
from mxnet import image
import numpy as np
import cv2
from utils.utils import load_pickle


class RandomSelectAug(image.Augmenter):
    """Randomly select one augmenter to apply, with chance to skip all.

    Parameters
    ----------
    aug_list : list of DetAugmenter
        The random selection will be applied to one of the augmenters
    skip_prob : float
        The probability to skip all augmenters and return input directly
    """

    def __init__(self, aug_list, skip_prob=0):
        super(RandomSelectAug, self).__init__(skip_prob=skip_prob)
        if not isinstance(aug_list, (list, tuple)):
            aug_list = [aug_list]
        if not aug_list:
            skip_prob = 1  # disabled

        self.aug_list = aug_list
        self.skip_prob = skip_prob

    def dumps(self):
        """Override default."""
        return [self.__class__.__name__.lower(), [x.dumps() for x in self.aug_list]]

    def __call__(self, src):
        """Augmenter implementation body"""
        if np.random.random() < self.skip_prob:
            return src
        else:
            np.random.shuffle(self.aug_list)
            return self.aug_list[0](src)


class CUHK03Loader(object):
    def __init__(self, path, P=5, K=5, transforms=None, img_mean=None, img_std=None):
        data_dict = load_pickle(path)

        self.train = data_dict['train']
        self.train_num = len(self.train)
        self.train_labels = np.arange(0, self.train_num, 1)
        np.random.shuffle(self.train_labels)
        self.train = self.train[self.train_labels]

        self.val = data_dict['val']
        self.val_labels = np.arange(0, len(self.val), 1)
        np.random.shuffle(self.val_labels)
        self.val = self.val[self.val_labels]
        self.val_labels = self.val_labels + self.train_num

        self.img_height = self.train[0][0].shape[0]
        self.img_weight = self.train[0][0].shape[1]
        self.img_channel = self.train[0][0].shape[2]

        self.P = P
        self.K = K

        if img_mean is None:
            self.img_mean = [0.485, 0.456, 0.406]  # RGB
        else:
            self.img_mean = img_mean
        if img_std is None:
            self.img_std = [0.229, 0.224, 0.225]  # RGB
        else:
            self.img_std = img_std

        self.img_std = nd.array(self.img_std).reshape(1, 1, 1, 3)
        self.img_mean = nd.array(self.img_mean).reshape(1, 1, 1, 3)

        self.base_transform = lambda img: (img / 255.0 - self.img_mean) / self.img_std
        if transforms is None:
            self.transforms = lambda img: img
        else:
            self.transforms = transforms

    def __len__(self):
        return self.val.shape[0]

    def train_loader(self):
        for i in range(len(self.train)):
            idx = np.random.choice(self.train.shape[0], size=self.P, replace=False)
            if not any(idx == i):
                idx[0] = i
            x = np.empty(shape=(self.P * self.K, self.img_height, self.img_weight, self.img_channel))
            y = np.empty(shape=self.P * self.K, dtype=np.int32)
            for n, j in enumerate(idx):
                persons = np.array(self.train[j])
                x[n * self.K:(n + 1) * self.K] = persons[
                    np.random.choice(persons.shape[0], size=self.K, replace=persons.shape[0] < self.K)]
                y[n * self.K:(n + 1) * self.K] = self.train_labels[j]
            # shuffle = np.arange(x.shape[0])
            # np.random.shuffle(shuffle)
            # x = x[shuffle]
            # y = y[shuffle]
            x = nd.array(x)
            y = nd.array(y)
            x = [self.transforms(x[i]) for i in range(x.shape[0])]
            x = nd.stack(*x, axis=0)
            x = self.base_transform(x)
            yield x.transpose(axes=(0, 3, 1, 2)), y

    def val_loader(self, p=None, k=None):
        if p is None:
            p = self.P
        if k is None:
            k = self.K

        for i in range(len(self.val)):
            idx = np.random.choice(self.val.shape[0], size=p, replace=False)
            idx[0] = i
            x = np.empty(shape=(p * k, self.img_height, self.img_weight, self.img_channel))
            y = np.empty(shape=p * k, dtype=np.int32)
            for n, j in enumerate(idx):
                persons = np.array(self.val[j])
                x[n * k:(n + 1) * k] = persons[np.random.choice(persons.shape[0], size=k, replace=persons.shape[0] < k)]
                y[n * k:(n + 1) * k] = self.val_labels[j]
            x = nd.array(x)
            x = self.base_transform(x)
            yield x.transpose(axes=(0, 3, 1, 2)), nd.array(y)

    def get_batch(self, p=None, k=None):
        if p is None:
            p = self.P
        if k is None:
            k = self.K

        i = np.random.choice(self.val.shape[0], size=1)[0]
        idx = np.random.choice(self.val.shape[0], size=p, replace=False)
        idx[0] = i
        x = np.empty(shape=(p * k, self.img_height, self.img_weight, self.img_channel))
        y = np.empty(shape=p * k, dtype=np.int32)
        for n, j in enumerate(idx):
            persons = np.array(self.val[j])
            x[n * k:(n + 1) * k] = persons[np.random.choice(persons.shape[0], size=k, replace=persons.shape[0] < k)]
            y[n * k:(n + 1) * k] = self.val_labels[j]
        im = x
        x = nd.array(x)
        x = self.base_transform(x)
        return x.transpose(axes=(0, 3, 1, 2)), nd.array(y), im


class DukeVideoReader:

    # Use
    # reader = DukeVideoReader('g:/dukemtmc/')
    # camera = 2
    # frame = 360720
    # img = reader.getFrame(camera, frame)

    def __init__(self, dataset_path):
        self.NumCameras = 8
        self.NumFrames = [359580, 360720, 355380, 374850, 366390, 344400, 337680, 353220]
        self.PartMaxFrame = 38370
        self.MaxPart = [9, 9, 9, 9, 9, 8, 8, 9]
        self.PartFrames = []
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 14250])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 15390])
        self.PartFrames.append([38400, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 10020])
        self.PartFrames.append([38670, 38670, 38670, 38670, 38670, 38700, 38670, 38670, 38670, 26790])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 21060])
        self.PartFrames.append([38400, 38370, 38370, 38370, 38400, 38400, 38370, 38370, 37350, 0])
        self.PartFrames.append([38790, 38640, 38460, 38610, 38760, 38760, 38790, 38490, 28380, 0])
        self.PartFrames.append([38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 38370, 7890])
        self.DatasetPath = dataset_path
        self.CurrentCamera = 1
        self.CurrentPart = 0
        self.PrevCamera = 1
        self.PrevFrame = -1
        self.PrevPart = 0
        self.Video = cv2.VideoCapture(
            '{:s}videos/camera{:d}/{:05d}.MTS'.format(self.DatasetPath, self.CurrentCamera, self.CurrentPart),
            cv2.CAP_FFMPEG)

    def getFrame(self, iCam, iFrame):
        # iFrame should be 1-indexed
        assert iFrame > 0 and iFrame <= self.NumFrames[iCam - 1], 'Frame out of range'
        # print('Frame: {0}'.format(iFrame))
        # Cam 4 77311
        # Coompute current frame and in which video part the frame belongs
        ksum = 0
        for k in range(10):
            ksumprev = ksum
            ksum += self.PartFrames[iCam - 1][k]
            if iFrame <= ksum:
                currentFrame = iFrame - 1 - ksumprev
                iPart = k
                break
        # Update VideoCapture object if we are reading from a different camera or video part
        if iPart != self.CurrentPart or iCam != self.CurrentCamera:
            self.CurrentCamera = iCam
            self.CurrentPart = iPart
            self.PrevFrame = -1
            self.Video = cv2.VideoCapture(
                '{:s}videos/camera{:d}/{:05d}.MTS'.format(self.DatasetPath, self.CurrentCamera, self.CurrentPart),
                cv2.CAP_FFMPEG)
        # Update time only if reading non-consecutive frames
        if not currentFrame == self.PrevFrame + 1:

            # if iCam == self.PrevCamera and iPart == self.PrevPart and currentFrame - self.PrevFrame < 30:
            #    # Skip consecutive images if less than 30 frames difference
            #    back_frame = max(self.PrevFrame, 0)
            # else:
            #    # Seek, but make sure a keyframe is read before decoding
            back_frame = max(currentFrame - 31, 0)  # Keyframes every 30 frames
            # print('back_frame set to: {0}'.format(back_frame))
            self.Video.set(cv2.CAP_PROP_POS_FRAMES, back_frame)
            if not self.Video.get(cv2.CAP_PROP_POS_FRAMES) == back_frame:
                print(
                    'Warning: OpenCV has failed to set back_frame to {0}. OpenCV value is {1}. Target value is {2}'.format(
                        back_frame, self.Video.get(cv2.CAP_PROP_POS_FRAMES), currentFrame))

            back_frame = self.Video.get(cv2.CAP_PROP_POS_FRAMES)
            # print('back_frame is: {0}'.format(back_frame))
            while back_frame < currentFrame:
                self.Video.read()
                back_frame += 1
        # print('currentFrame: {0}'.format(currentFrame))
        # print('current position: {0}'.format(self.Video.get(cv2.CAP_PROP_POS_FRAMES)))
        assert self.Video.get(cv2.CAP_PROP_POS_FRAMES) == currentFrame, 'Frame position error'
        result, img = self.Video.read()
        if result is False:
            print('-Could not read frame, trying again')
            back_frame = max(currentFrame - 61, 0)
            self.Video.set(cv2.CAP_PROP_POS_FRAMES, back_frame)
            if not self.Video.get(cv2.CAP_PROP_POS_FRAMES) == back_frame:
                print(
                    '-Warning: OpenCV has failed to set back_frame to {0}. OpenCV value is {1}. Target value is {2}'.format(
                        back_frame, self.Video.get(cv2.CAP_PROP_POS_FRAMES), currentFrame))
            back_frame = self.Video.get(cv2.CAP_PROP_POS_FRAMES)
            # print('-back_frame is: {0}'.format(back_frame))
            while back_frame < currentFrame:
                self.Video.read()
                back_frame += 1
            result, img = self.Video.read()

        img = img[:, :, ::-1]  # bgr to rgb
        # Update
        self.PrevFrame = currentFrame
        self.PrevCamera = iCam
        self.PrevPart = iPart
        return img


class DukeMTMCTVideo(object):
    def __init__(self, video_path, camera_id=1, bbox_path=None, gt_bbox_path=None, batch_size=100,
                 start_frame_pos=None):

        self.video_path = video_path
        self.batch_size = batch_size
        self.camera_id = camera_id
        self.video_reader = DukeVideoReader(self.video_path)
        self.cur_frames_idx = 1 if start_frame_pos is None else start_frame_pos
        # self.video_iter = self.get_video(self.video_path)
        # self.video, self.total_frames = next(self.video_iter)
        # self.pre_frame_count = 0
        # if start_frame_pos is not None:
        #     pre,cur = 0,self.total_frames
        #     while cur < start_frame_pos:
        #         self.video,self.total_frames = next(self.video_iter)
        #         pre,cur = cur,cur+self.total_frames
        #     self.video.set(cv2.CAP_PROP_POS_FRAMES,start_frame_pos-pre)
        #     self.pre_frame_count = pre
        self.fps = self.video_reader.Video.get(cv2.CAP_PROP_FPS)
        self.w = int(self.video_reader.Video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.video_reader.Video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if bbox_path is not None:
            self.bbox = load_pickle(bbox_path)
        if gt_bbox_path is not None:
            self.gt_bbox = load_pickle(os.path.join(gt_bbox_path, f'camera{camera_id}.pkl'))

    # @staticmethod
    # def get_video(video_path):
    #     for video_name in range(10):
    #
    #         video_name = f'{video_name:05d}.MTS'
    #         print('prepare:'+video_name)
    #         video = cv2.VideoCapture(os.path.join(video_path, video_name))
    #         assert video.isOpened(), 'video initial failed'
    #         total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    #         video.set(cv2.CAP_PROP_POS_FRAMES,total_frame-1)
    #         temp = video.read()
    #         while not temp[0]:
    #             total_frame -= 1
    #             video.set(cv2.CAP_PROP_POS_FRAMES, total_frame - 1)
    #             temp = video.read()
    #         video.set(cv2.CAP_PROP_POS_FRAMES,0)
    #         yield video, total_frame

    # yield video, video.get(cv2.CAP_PROP_FRAME_COUNT)

    def __iter__(self):
        return self

    def __next__(self):
        frames = []

        try:
            frames = [self.video_reader.getFrame(self.camera_id, i)
                      for i in range(self.cur_frames_idx, self.cur_frames_idx + self.batch_size, 1)]
            # cv2.imwrite('test.jpg',frames[0])
        except StopIteration:
            pass
        frame_pos_array = [i for i in range(self.cur_frames_idx, self.cur_frames_idx + len(frames), 1)]
        self.cur_frames_idx += len(frames)
        return_value = [frame_pos_array, frames]
        if hasattr(self, 'bbox'):
            bbox_list = [self.bbox.get(idx) for idx in frame_pos_array]
            return_value.append(bbox_list)
        if hasattr(self, 'gt_bbox'):
            gt_bbox_list = [self.gt_bbox.get(idx) for idx in frame_pos_array]
            return_value.append(gt_bbox_list)
        return tuple(return_value)


import os
import logging

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from gluoncv.data.base import VisionDataset


# modify from gluoncv.data.VOCDetection
class VOCDetection(VisionDataset):
    """Pascal VOC detection Dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indicies. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extreamly large.
    """
    CLASSES = DataConfig.detection_classes

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits=((2007, 'trainval'), (2012, 'trainval')),
                 transform=None, index_map=None, preload_label=True):
        super(VOCDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]


if __name__ == '__main__':

    # resize_aug = [
    #     image.ForceResizeAug(size=(270, 270)),
    #     image.RandomCropAug(size=(224,224))
    # ]
    # transforms = RandomSelectAug([
    #     image.HorizontalFlipAug(0.5),
    #     image.SequentialAug(resize_aug),
    #     image.BrightnessJitterAug(0.5)
    # ],0.2)
    # image.color_normalize()
    #
    #
    # a = CUHK03Loader(DataConfig.cuhk03_pkl_path)
    # for x,y in a.train_loader():
    #     print(x.shape)
    #     print(y.shape)

    video_stream = DukeMTMCTVideo('/data/stu06/MCMTDataSet/DukeMTMC/video/camera1',
                                  '/data/stu06/MCMTDataSet/DukeMTMC/bbox/det-camera1.pkl')
    # itor = video_stream.get_video()
    # for i in range(10):
    #     video_s, total_frames = next(itor)
    #     print(video_s.get(cv2.CAP_PROP_POS_FRAMES))
    #     print(total_frames)
    #     video_s.release()
    # p = np.random.random(size=(3651))
    # i = 0
    for last_frame_idx, frames, bboxes in video_stream:
        pass
        # if p[i] < 0.05 and bboxes[-1] is not None:
        #     print(last_frame_idx)
        #     for j in range(bboxes[-1].shape[0]):
        #         bbox = bboxes[-1][j]
        #         cv2.rectangle(frames[-1], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=123456)
        #     cv2.imwrite(f'../temp/test_{last_frame_idx}.jpg', frames[-1])
        # i += 1

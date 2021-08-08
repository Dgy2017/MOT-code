import os
import os.path as osp
import pickle as pkl
import time
from glob import glob

import PIL.Image as Image
import cv2
import h5py
import mxnet
import numpy as np
import scipy.io as sio
from lxml import etree as ET
from lxml.builder import ElementMaker
from mxnet import autograd
from mxnet import gluon
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from mxnet import nd

from configs.data_config import DataConfig


def market1501_to_pkl(val_ratio=0.2, market_root=DataConfig.market1501_root, replace=False):
    write_path = os.path.join(market_root, 'market1501.pkl')
    if os.path.exists(write_path) and not replace:
        print('file already exist!')
        return
    train_dir = os.path.join(market_root, 'bounding_box_train')
    test_dir = os.path.join(market_root, 'bounding_box_test')

    train_dict = dict()
    for img_file in tqdm(os.listdir(os.path.join(train_dir))):

        person_info = img_file.split('_')
        pid = person_info[0].strip()
        img = cv2.imread(os.path.join(train_dir, img_file))
        img = cv2.resize(img, dsize=(128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            train_dict[pid].append(img)
        except KeyError as e:
            train_dict[pid] = [img]

    val_dict = dict()
    for img_file in tqdm(os.listdir(os.path.join(test_dir))):

        person_info = img_file.split('_')
        pid = person_info[0].strip()
        img = cv2.imread(os.path.join(test_dir, img_file))
        img = cv2.resize(img, dsize=(128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            val_dict[pid].append(img)
        except KeyError as e:
            val_dict[pid] = [img]

    train, train_labels, val, val_labels = [], [], [], []
    for key, value in train_dict.items():
        train_labels.append(int(key))
        train.append(value)
    num = len(train)
    for key, value in val_dict.items():
        val.append(value)
        val_labels.append(int(key) + num)
    total_data = np.array(train + val)
    total_labels = np.array(train_labels + val_labels)

    sf = np.arange(0, len(total_data), step=1)
    np.random.shuffle(sf)
    separator = int(len(total_data) * val_ratio)
    val = total_data[sf[:separator]]
    val_labels = total_labels[sf[:separator]]

    train = total_data[sf[separator:]]
    train_labels = total_labels[sf[separator:]]

    print(f'{len(train)} ids for train,{len(val)} ids for validation')

    market_pkl = {
        'train': train,
        'train_labels': train_labels,
        'val': val,
        'val_labels': val_labels,
    }

    dump_pickle(market_pkl, write_path)


def cuhk03_to_pkl(part='labeled', read_path=DataConfig.cuhk03_mat_path,
                  write_path=DataConfig.cuhk03_pkl_path, replace=False,
                  val_ratio=0.2):
    if os.path.exists(write_path) and not replace:
        print('file already exist!')
        return
    assert part in ['labeled', 'detector']
    start = time.time()
    data = h5py.File(read_path, 'r')
    part = data[part][0]
    image_matrix = []
    for i, pair in enumerate(part):
        print(f'convert {i+1}th pair cameras...')
        pair = data[pair]
        person_image = [[] for _ in range(pair.shape[1])]
        for person in pair:
            for id, p in enumerate(tqdm(person)):
                img = data[p][:]
                if len(img.shape) == 3:
                    img = img.transpose(2, 1, 0)
                    img = cv2.resize(img, dsize=(128, 256))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    person_image[id].append(img)

        image_matrix += person_image
    image_matrix = np.array(image_matrix)
    sf = np.arange(0, len(image_matrix), step=1)
    np.random.shuffle(sf)
    separator = int(len(image_matrix) * val_ratio)
    val = image_matrix[sf[:separator]]
    val_labels = sf[:separator]

    train = image_matrix[sf[separator:]]
    train_labels = sf[separator:]

    cuhk03_pkl = {
        'train': train,
        'train_labels': train_labels,
        'val': val,
        'val_labels': val_labels
    }

    dump_pickle(cuhk03_pkl, write_path)


# copy from MASK_RCNN
def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


# copy from MASK_RCNN
def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    return an iou_matrix where the indices of (i,j) implies the IoU value of boxes1 with boxes2
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def nms_np(bbox, thresh):
    # bbox:(m,5)  thresh:scaler

    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = bbox[:, 4]
    keep = []

    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]  # because index start from 1

    return bbox[keep]


def det2pkl(path, use_nms=True, threshold=0.5):
    str2num = lambda x: [float(word) for word in x.strip().split(',')]
    res = dict()
    with open(path) as f:
        for sentence in tqdm(f.readlines()):
            sentence = str2num(sentence)
            try:
                res[int(sentence[1])].append(sentence[2:])
            except:
                res[int(sentence[1])] = [sentence[2:]]
    if use_nms:
        for key, value in tqdm(res.items()):
            res[key] = nms_np(np.array(res[key]), threshold)

    dump_pickle(res, path=path[:-3] + 'pkl')


def make_market1501_list(exdir, mode='train'):
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    train_dir = osp.join(exdir, "bounding_box_" + mode)
    train_list = {}
    for _, _, files in os.walk(train_dir, topdown=False):
        for name in files:
            if '.jpg' in name:
                name_split = name.split('_')
                pid = name_split[0]
                pcam = name_split[1][1]
                if pid not in train_list:
                    train_list[pid] = []
                train_list[pid].append({"name": name, "pid": pid, "pcam": pcam})

    with open(osp.join(exdir, mode + '.txt'), 'w') as f:
        for i, key in enumerate(train_list):
            for item in train_list[key]:
                f.write(item['name'] + " " + str(i) + " " + item["pcam"] + "\n")
    print("Make Label List Done")


def split_to_train_test(voc_root):
    assert osp.exists(voc_root), 'Voc root doesnt exit'

    train_file = osp.join(voc_root, 'ImageSets', 'Main', 'train.txt')
    test_file = osp.join(voc_root, 'ImageSets', 'Main', 'test.txt')
    xml_dir = osp.join(voc_root, 'Annotations')
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    ftrain = open(train_file, 'w')
    ftest = open(test_file, 'w')

    total_files = glob(osp.join(xml_dir, '*.xml'))
    total_files = [file.replace('\\', '/') for file in total_files]
    total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]

    train_files, test_files = train_test_split(total_files, test_size=0.3, random_state=42)

    # train
    for file in train_files:
        ftrain.write(file + "\n")

    # test
    for file in test_files:
        ftest.write(file + "\n")

    ftrain.close()
    ftest.close()

    print("split Completed. Number of Train Samples: {}. Number of Test Samples: {}".format(len(train_files),
                                                                                            len(test_files)))


def voc_format(img, img_name, bboxes, out_path, classes):
    """
    :param img:[W x H x 3] PIL format(RGB)
    :param bboxes:[N x (xmin,ymin,xmax,ymax)] the bounding boxes in this image
    :param out_path: the root path of the data
    :param classes: the class of each bounding box
    :return:
    """

    out_img_file = os.path.join(
        out_path, 'JPEGImages', img_name + '.jpg')
    out_xml_file = os.path.join(
        out_path, 'Annotations', img_name + '.xml')

    Image.fromarray(img).save(out_img_file)

    maker = ElementMaker()
    xml = maker.annotation(
        maker.folder(),
        maker.filename(img_name + '.jpg'),
        maker.source(
            maker.database(),  # e.g., The VOC2007 Database
            maker.annotation(),  # e.g., Pascal VOC2007
            maker.image(),  # e.g., flickr
        ),
        maker.size(
            maker.height(str(img.shape[0])),
            maker.width(str(img.shape[1])),
            maker.depth(str(img.shape[2])),
        ),
        maker.segmented(),
    )

    for bbox, class_name in zip(bboxes, classes):
        xml.append(
            maker.object(
                maker.name(class_name),  # 此处要改成英文的
                maker.pose(),
                maker.truncated(),
                maker.difficult(),
                maker.bndbox(
                    maker.xmin(str(bbox[0])),
                    maker.ymin(str(bbox[1])),
                    maker.xmax(str(bbox[2])),
                    maker.ymax(str(bbox[3])),
                ),
            )
        )

    # 保存标注文件XML
    with open(out_xml_file, 'wb') as f:
        f.write(ET.tostring(xml))


def get_new_color(n=1):
    color = np.random.randint(0, 256, size=(n, 3), dtype=np.int32)
    color = [tuple(i[0]) for i in np.vsplit(color, indices_or_sections=color.shape[0])]
    return color[0] if n == 1 else color


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v


def set_random_seed(seed):
    mxnet.random.seed(seed)
    np.random.seed(seed)


def get_logger():
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


def multi_gpu(model, inputs: list, ctx_list: list, batch_axis=0):
    with autograd.pause():
        if not isinstance(inputs, list):
            inputs = [inputs]
        new_inputs = [gluon.utils.split_and_load(x, ctx_list, batch_axis=batch_axis)
                      for x in inputs]

    return [model(*x) for x in zip(*new_inputs)]


def make_sure_dir(dir):
    dir = os.path.expanduser(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


def DukeMTMCTrackingGt2pkl(path, name):
    data = sio.loadmat(os.path.join(path, name))
    data = data['trainData']
    pkl_array = []
    for i in range(9):
        pkl_array.append(data[np.nonzero(data[:, 0] == (i + 1))])

    for i, pkl in enumerate(pkl_array):
        print(f'Converting Camera {i+1}...')
        pkl_dict = dict()
        camera, id, frame, coord = np.split(pkl, np.array([1, 2, 3]), axis=1)
        coord_id = np.concatenate([coord, id], axis=1)
        frame = np.squeeze(frame, axis=1)
        frame_set = list(set(frame))
        for f in tqdm(frame_set):
            indices, = np.nonzero(frame == f)
            pkl_dict[int(f)] = coord_id[indices]
        dump_pickle(pkl_dict, os.path.join(path, f'camera{i+1}.pkl'))


def test_from_directory(file_path, img_mean=None, img_std=None):
    # images should be named like this: label_name.jpg.
    # if not labels will be random assign to the images
    file_name = os.listdir(file_path)

    def parse_name(image_name: str):
        image_name = image_name.split('_')
        try:
            label = int(image_name[0])
            return label
        except:
            return None

    imgs = []
    labels = []
    label = parse_name(file_name[0])
    if label is not None:
        for name in file_name:
            labels.append(parse_name(name))
            img = cv2.imread(osp.join(file_path, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
    else:
        for name in file_name:
            labels.append(0)
            img = cv2.imread(osp.join(file_path, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
    imgs = np.array(imgs)

    if img_mean is None:
        img_mean = [0.485, 0.456, 0.406]  # RGB
    else:
        img_mean = img_mean
    if img_std is None:
        img_std = [0.229, 0.224, 0.225]  # RGB
    else:
        img_std = img_std

    img_std = nd.array(img_std).reshape(1, 1, 1, 3)
    img_mean = nd.array(img_mean).reshape(1, 1, 1, 3)
    base_transform = lambda img: (img / 255.0 - img_mean) / img_std
    x = nd.array(imgs)
    x = base_transform(x)
    return x.transpose(axes=(0, 3, 1, 2)), nd.array(labels), imgs,file_name


if __name__ == '__main__':
    market1501_to_pkl(replace=True)
    DukeMTMCTrackingGt2pkl(data_cfg.gt_path,'trainval.mat')
    # cuhk03_to_pkl()
    # for i in range(1):
    #     det2pkl(f'/data/stu06/MCMTDataSet/DukeMTMC/bbox/det-camera{i+1}.txt',threshold=0.3)

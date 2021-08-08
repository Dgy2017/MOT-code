import cv2
import motmetrics as mm
import mxnet as mx
import numpy as np
from mxnet import nd

from configs.pipeline_config import PipelineConfig as pl_cfg
from models.loss import euler_distance
from utils.utils import compute_overlaps
from utils.utils import get_new_color


def check_end(trajectories):
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    for tra in trajectories:
            tra.auto_set_end()


def make_sure_order(tra1, tra2):
    check_end([tra1, tra2])
    assert tra1.end < tra2.start or tra2.end < tra1.start, 'There is an overlap between two trajectories'
    if tra1.end < tra2.end:
        return tra1, tra2
    else:
        return tra2, tra1


class FrameInfo(object):
    def __init__(self, trajectory_array, picture, bboxes, features, frame_idx, gt_bbox):
        self.trajectory_array = trajectory_array
        self.picture = picture
        self.bboxes = bboxes
        self.features = features
        self.frame_idx = frame_idx
        self.gt_bbox = gt_bbox


class Galley(object):
    def __init__(self, video_name, k=5, cvtMode=None, motMetrics: mm.MOTAccumulator = None,fps=60,w=1920,h=1080,version = 1):
        self.video_writer = None
        self.video_name = video_name
        self.video_count = 0

        self.gallery_size = k
        self.pointer = 0
        self.cvtMode = cvtMode
        self.motMetrics = motMetrics
        self.version = version

        self.frame_count = 0
        self.w = w
        self.h = h
        self.fps = fps
        self.gallery = np.array([None] * self.gallery_size)
        self.update_video_writer()

    def update_video_writer(self):
        self.frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.video_name + str(self.video_count) + '.avi',
                                            fourcc, self.fps,(self.w, self.h))
        self.video_count += 1

    def add_frame_info(self, frame_info):

        if self.pointer < self.gallery_size:
            self.gallery[self.pointer] = frame_info
            self.pointer += 1
        else:
            self.write_video(self.gallery[self.pointer % self.gallery_size])
            self.gallery[self.pointer % self.gallery_size] = frame_info
            self.pointer += 1

    def write_video(self, frame_info: FrameInfo):
        frame = frame_info.picture
        if self.version == 2:
            if frame_info.trajectory_array is not None:
                for i, item in enumerate(frame_info.trajectory_array):
                    if item.parent is not None:
                        dist = item.tra_distance(item.parent)
                        l, r, sigma, u = item.parent.ideal_region(2.5, detail=True)
                        if dist < r:
                            item.merge_trajectory(item.parent)
                            item.parent = None

        o_bbox = frame_info.gt_bbox[:, 0:4] if frame_info.gt_bbox is not None else []
        o_id = frame_info.gt_bbox[:, -1] if frame_info.gt_bbox is not None else []

        h_bbox = []
        h_id = []
        dist_matrix = []
        if self.cvtMode is not None:
            frame = cv2.cvtColor(frame, self.cvtMode)
        if frame_info.bboxes is not None:
            tra_filter = (frame_info.trajectory_array > pl_cfg.tra_length_low_bound)
            frame_info.trajectory_array = frame_info.trajectory_array[tra_filter]
            frame_info.bboxes = frame_info.bboxes[tra_filter]
            bbox_list = frame_info.bboxes
            color_list = [tra.color for tra in frame_info.trajectory_array]
            h_id = [tra.id for tra in frame_info.trajectory_array]

            for b, c, tid in zip(list(bbox_list), color_list, h_id):
                #todo delete
                frame = cv2.rectangle(frame, tuple(b[0:2]), tuple(b[2:4]), color=[int(cc) for cc in c], thickness=3)
                frame = cv2.putText(frame, f'Tid:{tid}', (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    color=[int(cc) for cc in c], thickness=2)
            h_bbox = bbox_list[:, :4]
            h_bbox[:, (2, 3)] = h_bbox[:, (2, 3)] - h_bbox[:, (0, 1)]
            dist_matrix = mm.distances.iou_matrix(o_bbox, h_bbox, max_iou=0.5)

        self.motMetrics.update(o_id, h_id, dist_matrix)
        if self.frame_count >= 38370:
            self.update_video_writer()
        self.video_writer.write(frame)
        self.frame_count += 1

    def reverse_slice(self, step=1):
        start_idx = max(self.pointer - 2, -1)
        end_idx = max(self.pointer - self.gallery_size - 1, -1)
        step = step if step > 0 else -step
        idx = [i % self.gallery_size for i in range(start_idx, end_idx, -step)]
        return self.gallery[idx]

    def clear_gallery(self):
        assert self.pointer >= self.gallery_size, 'The gallery did not fill up'
        for i in range(self.gallery_size):
            self.write_video(self.gallery[(self.pointer + i) % self.gallery_size])
        self.video_writer.release()

    def top(self) -> FrameInfo:
        return self.gallery[(self.pointer - 1) % self.gallery_size] if self.pointer > 0 else None


class Trajectory(object):
    total = 0
    context = mx.cpu()

    def __init__(self, people: nd.NDArray, bbox: np.ndarray, frame_idx: int,parent=None):
        assert isinstance(people, nd.NDArray), 'people:please use nd.NDArray'
        assert isinstance(bbox, np.ndarray), 'bbox:please use np.ndarray'
        assert people is not None
        people = people.as_in_context(context=Trajectory.context)

        self.bboxes = [(bbox, frame_idx)]
        self.features = [people]
        self.feat_sum = people.copy()
        self.feat_n = 1
        self.id = Trajectory.total
        Trajectory.total += 1
        self.color = get_new_color()
        self.dist_square_sum = nd.zeros(shape=1)
        self.dist_sum = nd.zeros(shape=1)
        self.dist_n = nd.zeros(shape=1)
        self.start = frame_idx
        self.end = None
        self.parent = parent

    def __lt__(self, other):
        return self.feat_n < other

    def __gt__(self, other):
        return self.feat_n > other

    def set_end(self, frame_idx):
        self.end = frame_idx

    def set_start(self, frame_idx):
        self.start = frame_idx

    def add_people(self, people, bbox, frame_idx):

        people = people.as_in_context(Trajectory.context)
        self.features.append(people)
        self.feat_sum += people
        self.feat_n += 1

        self.bboxes.append((bbox, frame_idx))

    def auto_set_end(self):
        self.end = self.bboxes[-1][1]

    def merge_trajectory(self, tra):
        check_end([self, tra])
        assert self.start > tra.end, '仅可以合并早于改轨迹的轨迹'
        tra.features.extend(self.features)
        self.features = tra.features
        self.feat_sum += tra.feat_sum
        self.feat_n += tra.feat_n

        self.dist_square_sum = tra.dist_square_sum + self.dist_square_sum
        self.dist_sum += tra.dist_sum
        self.dist_n += tra.dist_n

        tra.bboxes.extend(self.bboxes)
        self.bboxes = tra.bboxes

    def ideal_region(self, n=1.5, detail=False):
        u = nd.clip(self.dist_sum / self.dist_n,
                    a_min=1e-12, a_max=1e12)
        sigma = nd.sqrt(nd.clip((self.dist_square_sum - self.dist_n * nd.square(u)) / (self.dist_n - 1),
                                a_min=1e-12, a_max=1e12))

        if detail:
            return u - n * sigma, u + n * sigma, sigma, u
        else:
            return u - n * sigma, u + n * sigma

    def add_dist(self, dist):
        self.dist_square_sum += nd.sum(nd.square(dist))
        self.dist_sum += nd.sum(dist)
        self.dist_n += dist.shape[0]

    def bbox_distance(self, old_frame_idx, people: nd.NDArray, bboxes: np.ndarray, frame_idx: int,
                      sample_num=4) -> np.ndarray:
        '''

        :param people: 一系列人物特征的集合[N x 2048]
        :param bbox: 对应的边界框[N x 5]
        :param frame_idx:对应的帧[1](因为这个一系列人来自同一帧)
        :param sample_num:在轨迹种选取人物的个数
        :return:[1 x N]表示该轨迹与N个bbox的距离
        '''
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)
        if isinstance(people, list):
            people = nd.stack(*people)
        last_frame_idx = self.bboxes[-1][1]
        bbox_num = len(self.bboxes)
        search_head = max(0, bbox_num - last_frame_idx + old_frame_idx)
        while search_head < bbox_num and self.bboxes[search_head - 1][1] != old_frame_idx:
            search_head += 1

        if sample_num == -1:
            sample_num = search_head
        sample_num = sample_num if sample_num <= search_head else search_head
        temp = search_head - sample_num
        tra_frame = np.array([item[1] for item in self.bboxes[temp:search_head]])
        tra_bboxes = np.array([item[0] for item in self.bboxes[temp:search_head]])
        tra_people = nd.stack(*self.features[temp:search_head])

        iou_matrix = compute_overlaps(tra_bboxes, bboxes)
        dist_matrix = euler_distance(tra_people, people)
        # todo 调整alpha，beta值，当前值不合适
        # threshold = np.exp(pl_cfg.threshold_alpha / (frame_idx - tra_frame) - pl_cfg.threshold_beta)
        threshold = 0.05 * (tra_frame - frame_idx) + 0.6
        threshold = threshold.reshape(threshold.shape[0], 1)
        dist = pl_cfg.s - dist_matrix.asnumpy() #- np.where(iou_matrix < threshold, pl_cfg.inf, 0)
        # dist = pl_cfg.s - dist_matrix.asnumpy()# - np.where(iou_matrix < threshold, pl_cfg.inf, 0)
        return np.mean(dist, axis=0)

    # def feat_distance(self, people: nd.NDArray, sample_num=5):
    #     assert isinstance(people, nd.NDArray), 'please use nd.NDArray'
    #     if sample_num == -1:
    #         sample_num = len(self.features)
    #     sample_num = sample_num if sample_num <= len(self.features) else len(self.features)
    #
    #     dist = nd.norm(nd.stack(*self.features[len(self.features) - sample_num:], axis=0) -
    #                    people.expand_dims(0),
    #                    ord=2,
    #                    axis=1)
    #     return dist
    def feat_distance(self, people: nd.NDArray, sample_num=5):
        '''

        :param people: M x N
        :param sample_num:
        :return:
        '''
        if len(people.shape)==1:
            people = people.expand_dims(0)
        assert isinstance(people, nd.NDArray), 'please use nd.NDArray'
        if sample_num == -1:
            sample_num = len(self.features)
        sample_num = sample_num if sample_num <= len(self.features) else len(self.features)
        n = len(self.features)
        step = np.minimum(n // sample_num,50)
        idx = [n - i*step - 1 for i in range(sample_num)]

        dist = nd.norm(nd.stack(*[self.features[index] for index in idx], axis=0).expand_dims(0) -
                       people.expand_dims(1),
                       ord=2,
                       axis=2)
        return nd.squeeze(nd.mean(dist,axis=1))

    def get_feat_mean(self):
        return self.feat_sum / self.feat_n

    def tra_distance(self, tra):
        check_end([self, tra])
        interval = nd.maximum(self.start - tra.end, tra.start - self.end)
        if not(0 < interval < pl_cfg.max_interval):
            print(f'self-start {self.start},self-end {self.end},tra-start {tra.start},tra-end {tra.end}')
        return nd.norm(self.get_feat_mean() - tra.get_feat_mean(), ord=2) + \
               0 if 0 < interval < pl_cfg.max_interval else pl_cfg.inf

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


if __name__ == '__main__':
    sample_num = 200
    feature = nd.random_normal(0.5, scale=0.2, shape=(sample_num, 2048))
    bboxes = np.random.randint(low=0, high=1024, size=(sample_num, 5))
    frame_idx = np.random.choice(sample_num, size=sample_num, replace=False)
    frame_idx = np.sort(frame_idx)
    tra2_start = int(sample_num // 2)
    tra1 = Trajectory(feature[0], bboxes[0], frame_idx[0])
    tra2 = Trajectory(feature[tra2_start], bboxes[tra2_start], frame_idx[tra2_start])
    total_dist = []
    for i in range(1, tra2_start, 1):
        dist = tra1.feat_distance(people=feature[i:4])
        tra1.add_people(feature[i], bboxes[i], frame_idx[i])
        tra1.add_dist(dist)
        total_dist.append(dist)
    total_dist = nd.concatenate(total_dist, axis=0)
    total_dist = total_dist.asnumpy()

    l, r, sigma, u = tra1.ideal_region(1.5, detail=True)

    real_sigma = np.sqrt(np.var(total_dist))
    feat_mean = tra1.get_feat_mean()
    real_mean = nd.mean(feature[:tra2_start], axis=0)
    delta = feat_mean - real_mean
    total_dist2 = []
    for i in range(1, tra2_start, 1):
        dist2 = tra2.feat_distance(people=feature[i + tra2_start])
        tra2.add_people(feature[i + tra2_start], bboxes[i + tra2_start], frame_idx[i + tra2_start])
        tra2.add_dist(dist2)
        total_dist2.append(dist2)
    # tra2_feat_mean = tra2.get_feat_mean()
    # l2,r2,sigma2,u2=tra2.ideal_region(1.5,detail=True)

    total_dist2 = nd.concatenate(total_dist2, axis=0)
    total_dist2 = total_dist2.asnumpy()
    # real_sigma2 = np.var(total_dist2)
    # # tra2.merge_trajectory(tra1)
    # tra1.merge_trajectory(tra2)
    # t_feat_mean = tra1.get_feat_mean()
    # t_real_mean = nd.mean(feature,axis=0)
    # t_l,t_r,t_sigma,t_u=tra1.ideal_region(1.5,detail=True)
    # t_real_dist_sigma=np.var(np.concatenate((total_dist,total_dist2),axis=0))
    # print('debug')

    # t_people = feature[tra2_start:tra2_start+10]
    # t_bboxes = bboxes[tra2_start:tra2_start+10]
    # frame_idx = 100
    # dist = tra1.bbox_distance(t_people,t_bboxes,frame_idx)
    # print('d')

    # tra1.merge_trajectory(tra2)
    # merge_total_dist1 = np.stack([total_dist, total_dist2], axis=0)
    # real_dist_sigma3 = np.sqrt(np.var(merge_total_dist1))
    #
    # l, r, sigma, u = tra1.ideal_region(1.5, detail=True)
    # feat_mean = tra1.get_feat_mean()
    # real_feat_mean = nd.mean(feature,axis=0)
    # delta = feat_mean - real_feat_mean
    # print('d')
    a = np.array([tra1, tra2], copy=False)
    # b = np.array([0,1])
    # tra3 = a[b<1]
    # tra3[0].merge_trajectory(tra2)
    print(a < 3)

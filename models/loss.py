import numpy as np
from mxnet import ndarray as nd
from mxnet.gluon.loss import _apply_weighting

from configs.alignedreid_config import AlignedReIdConfig


def euler_distance(x: nd.NDArray, y: nd.NDArray):
    assert x.shape[1] == y.shape[1], f"x.shape[1] does'nt match y.shape[1], " \
                                     f"x.shape[1]={x.shape[1]} while y.shpae[1]={y.shape[1]}"

    x = x.expand_dims(axis=1)
    y = y.expand_dims(axis=0)

    dist = nd.sqrt(nd.clip(nd.sum(nd.square(x - y), axis=2), a_min=1e-12, a_max=1e12))

    return dist


def get_correspond_entity(feature, labels, normalize=True):
    if normalize:
        feature = feature / (nd.norm(feature, axis=1, keepdims=True) + 1e-12)
    dist_mat = euler_distance(feature, feature)
    for i in range(dist_mat.shape[0]):
        dist_mat[i, i] = 99999
    pred = nd.argmin(dist_mat, axis=1)
    return labels[pred]


def weighted_samples_loss(dist_mat: nd.NDArray, labels):
    assert len(dist_mat.shape) == 2
    assert dist_mat.shape[0] == dist_mat.shape[1]

    pos_idx = (labels.expand_dims(1) == labels.expand_dims(0)).astype(np.float32)
    pos_idx = pos_idx.as_in_context(dist_mat.context)
    neg_idx = 1 - pos_idx
    filler = nd.ones_like(dist_mat) * (-1e12)
    filler = filler.as_in_context(dist_mat.context)
    w_pos = nd.softmax(nd.where(pos_idx, dist_mat, filler), axis=1)
    w_neg = nd.softmax(nd.where(neg_idx, -dist_mat, filler), axis=1)

    dist_ap = nd.sum(dist_mat * w_pos, axis=1)
    dist_an = nd.sum(dist_mat * w_neg, axis=1)

    return dist_ap, dist_an


def hardest_samples_loss(dist_mat, labels):
    assert len(dist_mat.shape) == 2
    assert dist_mat.shape[0] == dist_mat.shape[1]

    pos_idx = (labels.expand_dims(1) == labels.expand_dims(0)).astype(np.float32)
    neg_idx = 1 - pos_idx

    filler = (nd.ones_like(dist_mat) * nd.max(dist_mat)).as_in_context(dist_mat.context)
    dist_ap = nd.max(nd.where(pos_idx, dist_mat, -filler), axis=1)
    dist_an = nd.min(nd.where(neg_idx, dist_mat, filler), axis=1)
    # dist_ap = nd.sum(pos_idx*dist_mat,axis=1)/nd.sum(pos_idx,axis=1)
    # dist_an = nd.sum(neg_idx*dist_mat,axis=1)/nd.sum(neg_idx,axis=1)

    return dist_ap, dist_an


def local_dist_mat(x, y, shape):
    dist_mat = euler_distance(x, y)
    dist_mat = dist_mat.exp()
    dist_mat = (dist_mat - 1.0) / (dist_mat + 1.0)
    dist_mat = dist_mat.reshape(*shape)
    dist_mat = dist_mat.transpose(axes=(1, 3, 0, 2))
    m, n = shape[1], shape[3]
    S = [[0 for _ in range(m)] for __ in range(n)]
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                S[i][j] = dist_mat[i, j]
            elif i == 1:
                S[i][j] = S[i - 1][j]
            elif j == 1:
                S[i][j] = S[i][j - 1]
            else:
                S[i][j] = nd.minimum(S[i - 1][j], S[i][j - 1]) + dist_mat[i, j]
    return S[-1][-1]


class Loss(object):
    """Base class for loss. modified from gluon source code

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """

    def __init__(self, weight, batch_axis, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def __repr__(self):
        s = '{name}(batch_axis={_batch_axis}, w={_weight})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LocalLoss(Loss):

    def __init__(self, margin=0.5, weight=None, mode='hardest', batch_axis=0,dist_mat=False, **kwargs):
        super(LocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin
        self.dist_mat = dist_mat
        self.mode=mode
        assert mode.lower() in ['hardest', 'weighted']
        if mode.lower() == 'hardest':
            self.sample_function = hardest_samples_loss
        else:
            self.sample_function = weighted_samples_loss

    def __call__(self, x, labels):
        shape = x.shape[:2]
        x = x.reshape(-1, x.shape[2])
        # x = 1.0 * x/(nd.norm(x,ord=2,axis=1,keepdims=True)+1e-12)
        dist_mat = local_dist_mat(x, x, shape=(*shape, *shape))
        dist_ap, dist_an = self.sample_function(dist_mat, labels)

        loss = nd.relu(self._margin + dist_ap - dist_an)
        if not self.dist_mat:
            return nd.mean(_apply_weighting(nd, loss, self._weight)), dist_ap, dist_an
        else:
            return nd.mean(_apply_weighting(nd, loss, self._weight)), dist_ap, dist_an,dist_mat

class GlobalLoss(Loss):
    def __init__(self, margin=0.5, weight=None, mode='hardest', batch_axis=0, dist_mat = False, **kwargs):
        super(GlobalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin
        self.dist_mat = dist_mat
        self.mode = mode.lower()
        assert mode.lower() in ['hardest', 'weighted']
        if mode.lower() == 'hardest':
            self.sample_function = hardest_samples_loss
        else:
            self.sample_function = weighted_samples_loss

    def __call__(self, feature, labels):
        feature = feature / (nd.norm(feature, axis=1, keepdims=True, ord=2) + 1e-12)

        dist_mat = euler_distance(feature, feature)
        dist_ap, dist_an = self.sample_function(dist_mat, labels)

        loss = nd.relu(dist_ap - dist_an + self._margin)
        # loss  = dist_ap - dist_an
        if not self.dist_mat:
            return nd.mean(_apply_weighting(nd, loss, self._weight)), dist_ap, dist_an
        else:
            return nd.mean(_apply_weighting(nd, loss, self._weight)), dist_ap, dist_an,dist_mat


# def test_euler_distance(x, y):
#     dist_mat = nd.empty(shape=(x.shape[0], y.shape[0]))
#     for i in range(x.shape[0]):
#         for j in range(y.shape[0]):
#             dist_mat[i, j] = nd.sqrt(nd.clip(nd.sum(nd.square(x[i] - y[j])), a_min=1e-12, a_max=1e12))
#     return dist_mat


if __name__ == '__main__':
    # x = nd.random_uniform(low=-100,high=100,shape=[10,5])
    # y = nd.random_uniform(low=-100,high=100,shape=[5,5])
    # dist = euler_distance(x, x)
    # dist_test = test_euler_distance(x,x)
    # print(dist==dist_test)
    #
    #
    # labels = nd.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
    # dist = dist.as_in_context(mx.gpu())
    # labels = labels.as_in_context(mx.gpu())
    # hardest_samples_loss(dist,labels)

    from mxnet.gluon import nn
    from mxnet import autograd

    model = nn.Sequential()
    model.add(nn.Conv2D(128, 1, 1, 0))
    model.add(nn.AvgPool2D((1, 7)))
    model.initialize()

    loss = LocalLoss(margin=0.5, weight=1, mode='weighted')
    input = nd.random_uniform(0, 1, shape=(10, 1, 7, 7))
    labels = nd.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
    model.collect_params()
    with autograd.record(True):
        out = model(input)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = out.swapaxes(1, 2)
        l, dist_ap, dist_an = loss(out, labels)
        # l[1,0].backward()
        l.backward()
    print('')
    # a = nd.arange(0, 12, step=1).reshape(2, 3, 2)
    #
    # b = nd.arange(-1,11,step=1).reshape(2, 3, 2)
    # test_dist = test_(a,a)
    # #print(a)
    # a = a.reshape(-1,2)
    # #print(a)
    # b = b.reshape(-1,2)
    # dist = euler_distance(a,a)
    # dist = dist.reshape(2,3,2,3)
    #
    #
    # print(dist==test_dist)

    # model = nn.Sequential()
    # model.add(nn.Conv2D(3,1,1,0))
    # model.add(nn.AvgPool2D((1,7)))
    # model.initialize()
    # input = nd.random_uniform(0,1,shape=(10,1,7,7))
    # loss_function = LocalLoss()
    # labels = nd.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
    # with autograd.record(True):
    #     out = model(input)
    #     out = out.reshape(10, 3, 7)
    #     out = out.swapaxes(1,2)
    #     loss = loss_function(out,labels)
    # # out.backward()
    # loss.backward()
    # print('')

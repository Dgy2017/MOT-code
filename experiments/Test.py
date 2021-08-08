import sys

sys.path.append('../')
import numpy as np
import cv2
from configs.alignedreid_config import AlignedReIdConfig
from configs.data_config import DataConfig
from dataloader.dataloader import CUHK03Loader
from models.resnet import resnet50_v1
from models.loss import GlobalLoss, LocalLoss, euler_distance
import utils.utils as utils
from mxnet import nd
import os
from mxnet import cpu
from experiments.metric import AvgMetric



def get_visual_result():
    utils.set_random_seed(AlignedReIdConfig.random_seed)


    data_loader = CUHK03Loader(os.path.join(DataConfig.market1501_root, 'market1501.pkl'),
                               AlignedReIdConfig.batch_p,
                               AlignedReIdConfig.batch_k,
                               transforms=AlignedReIdConfig.transforms)
    model = resnet50_v1(ctx=AlignedReIdConfig.ctx, pretrained=False, use_fc=AlignedReIdConfig.use_fc,
                        classes=data_loader.train_num)
    model.load_parameters(os.path.join(AlignedReIdConfig.check_point, 'model_0.878.params'),
                          ctx=AlignedReIdConfig.ctx)
    g_loss_fun = GlobalLoss(AlignedReIdConfig.global_margin, AlignedReIdConfig.global_loss_weight,
                            AlignedReIdConfig.global_loss_mode)
    l_loss_fun = LocalLoss(AlignedReIdConfig.local_margin, AlignedReIdConfig.local_loss_weight,
                           AlignedReIdConfig.local_loss_mode)

    if AlignedReIdConfig.test_load_right:
        # val metric
        val_acc = AvgMetric('V-Acc')
        val_g_ap = AvgMetric('V-g_ap')
        val_g_an = AvgMetric('V-g_an')
        val_l_ap = AvgMetric('V-l_ap')
        val_l_an = AvgMetric('V-l_an')
        val_loss = AvgMetric('V-Loss')
        for imgs, labels in data_loader.val_loader(40, 2):
            imgs = imgs.as_in_context(AlignedReIdConfig.ctx)
            labels = labels.as_in_context(AlignedReIdConfig.ctx)
            g_feature, l_feature = model(imgs)
            g_loss, g_ap, g_an = g_loss_fun(g_feature, labels)
            l_loss, l_ap, l_an = l_loss_fun(l_feature, labels)
            loss = g_loss + l_loss
            acc = nd.sum(g_ap < g_an) / len(g_ap)
            val_acc.update(acc)
            val_g_an.update(nd.mean(g_an).asscalar())
            val_g_ap.update(nd.mean(g_ap).asscalar())
            val_l_an.update(nd.mean(l_an).asscalar())
            val_l_ap.update(nd.mean(l_ap).asscalar())
            val_loss.update(loss.asscalar())

        print('{}:{:.4f}'.format(*val_loss.get()) + \
              '{}:{:.4f}'.format(*val_acc.get()) + \
              '{}:{:.4f}'.format(*val_g_ap.get()) + \
              '{}:{:.4f}'.format(*val_g_an.get()) + \
              '{}:{:.4f}'.format(*val_l_ap.get()) + \
              '{}:{:.4f}'.format(*val_l_an.get())
              )
    x, labels, imgs,file_names = utils.test_from_directory('/data/stu06/homedir/project/gluon/MCMT/experiments/reid_test_imgs')
    x = x.as_in_context(AlignedReIdConfig.ctx)
    labels = labels.as_in_context(AlignedReIdConfig.ctx)
    g_feature, l_feature,_ = model(x)
    g_feature = g_feature / nd.norm(g_feature, ord=2, axis=1, keepdims=True)
    dist_mat = euler_distance(g_feature, g_feature)
    print(file_names)
    print(dist_mat)
    for i in range(x.shape[0]):
        img = imgs[i]
        img = cv2.resize(img, dsize=(128, 224))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        label = labels[i].asscalar()
        dist = dist_mat[0, i].asscalar()
        # print(dist_mat[5:10,5:10])
        cv2.imwrite(os.path.join(AlignedReIdConfig.image_dir, f'{file_names[i]}_{label}_{i}_{dist:.4}.jpg'),
                    img)


if __name__ == '__main__':
    get_visual_result()

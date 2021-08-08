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
from tqdm import tqdm

def parameters_mean(target_param,parameters):
    n = len(parameters)
    parameter_res = parameters.pop()
    for para in parameters:
        for d1,d2 in zip(parameter_res._data,para._data):
            d1 += d2
    for target_data,data in zip(target_param._data,parameter_res._data):
        target_data[:] = (data / n)[:]



def weighted_parameters(model_constructor, model_names, classes=None):
    model_res = resnet50_v1( pretrained=False, use_fc=AlignedReIdConfig.use_fc,
                        classes=classes)
    model_res.load_parameters(os.path.join(AlignedReIdConfig.check_point, model_names[0]),
                          ctx=cpu())
    model_list = []
    for name in model_names:

        model = resnet50_v1(ctx=cpu(), pretrained=False, use_fc=AlignedReIdConfig.use_fc,
                        classes=classes)
        model.load_parameters(os.path.join(AlignedReIdConfig.check_point, name),
                              ctx=cpu())
        model_list.append(model)

    res_ParameterDict = model_res.collect_params()
    prefix = res_ParameterDict.prefix
    params = res_ParameterDict._params

    other_parameter_dicts = [model.collect_params() for model in model_list]
    other_prefix = [parameter_dict.prefix for parameter_dict in other_parameter_dicts]
    other_params = [parameter_dict._params for parameter_dict in other_parameter_dicts]
    for key in params.keys():
        # params[key] = parameters_mean([o_params[key.replace(prefix,o_prefix)] for o_prefix,o_params in zip(other_prefix,other_params)])
        parameters_mean(params[key],
            [o_params[key.replace(prefix, o_prefix)] for o_prefix, o_params in zip(other_prefix, other_params)])

    res_ParameterDict.reset_ctx(AlignedReIdConfig.ctx)
    model_res.collect_params()
    return model_res









def get_visual_result():
    utils.set_random_seed(AlignedReIdConfig.random_seed)


    data_loader = CUHK03Loader(os.path.join(DataConfig.market1501_root, 'market1501.pkl'),
                               AlignedReIdConfig.batch_p,
                               AlignedReIdConfig.batch_k,
                               transforms=AlignedReIdConfig.transforms)

    # model = weighted_parameters(resnet50_v1, AlignedReIdConfig.weighted_models,classes=data_loader.train_num)
    model = resnet50_v1(ctx=AlignedReIdConfig.ctx, pretrained=False, use_fc=AlignedReIdConfig.use_fc,
                        classes=data_loader.train_num)
    model.load_parameters(os.path.join(AlignedReIdConfig.check_point, 'model_0.8197_margin_1.params'),
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
        for imgs, labels in tqdm(data_loader.val_loader(12, 4)):
            imgs = imgs.as_in_context(AlignedReIdConfig.ctx)
            labels = labels.as_in_context(AlignedReIdConfig.ctx)
            g_feature, l_feature,_ = model(imgs)
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



if __name__ == '__main__':
    get_visual_result()

from utils import utils
from configs.alignedreid_config import AlignedReIdConfig as reid_cfg
import numpy as np
import os.path as osp
from configs.data_config import DataConfig as data_cfg
from dataloader.dataloader import CUHK03Loader
import os
import matplotlib.pyplot as plt
from mxnet import nd
# training_data = CUHK03Loader(os.path.join(data_cfg.market1501_root, 'market1501.pkl'),
#                                reid_cfg.batch_p,
#                                reid_cfg.batch_k,
#                                transforms=reid_cfg.transforms)

training_info = utils.load_pickle(osp.join(reid_cfg.training_info,'training_info.pkl'))
# weighted_training_info = utils.load_pickle(osp.join(reid_cfg.training_info,'weighted_training_info.pkl'))
mutual_training_info= utils.load_pickle(osp.join(reid_cfg.training_info,'mutual_training_info.pkl'))
#todo

def clear(dict_list):
    from experiments.metric import AvgMetric
    for d in dict_list:
        for key,value in d.items():
            if isinstance(value,AvgMetric):
                if isinstance(value.value_list[0],nd.NDArray):
                    value.value_list = [value.asscalar() for value in value.value_list]
                    d[key]=value
    return dict_list


def mean_(value_list,delimiters:list):
    value_list = value_list[:delimiters[-1]]
    if isinstance(value_list[0],nd.NDArray):
        value_list = [item.asscalar() for item in value_list]
    value_list = np.array(value_list)
    value_list = np.split(value_list,len(value_list)/delimiters[0])

    value_list = np.array([np.mean(value) for value in value_list])
    return value_list

def get_max(a:np.array):
    m = -np.inf
    for i in range(len(a)):
        if a[i]>m:
            m = a[i]
        else:
            a[i]=m
    return a

def weighted_or_not():
    delimiters = weighted_training_info['train_acc'].delimiters \
    if len(training_info['train_acc'].delimiters)>len(weighted_training_info['train_acc'].delimiters) else training_info['train_acc'].delimiters
    delimiters.pop(0)
    delimiters = delimiters[:-3]
    # delimiters = np.array(delimiters)+1
    # delimiters = np.arange(3000)
    normal_acc=mean_(training_info['train_acc'].value_list,delimiters)
    normal_v_acc=mean_(training_info['val_acc'].value_list,delimiters)
    weighted_acc = mean_(weighted_training_info['train_acc'].value_list, delimiters)
    weighted_v_acc = mean_(weighted_training_info['val_acc'].value_list, delimiters)

    normal_g_ap = mean_(training_info['train_g_ap'].value_list,delimiters)
    normal_g_an = mean_(training_info['train_g_an'].value_list,delimiters)

    weighted_g_ap = mean_(weighted_training_info['train_g_ap'].value_list,delimiters)
    weighted_g_an = mean_(weighted_training_info['train_g_an'].value_list,delimiters)

    # fig,(ax1,ax2)=plt.subplots(1,2)
    x = np.arange(len(normal_acc))+1
    # plt.plot(x,get_max(normal_acc),label='TriHard Train Acc', color='green', linestyle='dashed')
    # plt.plot(x, get_max(weighted_acc), label='TriWeighted Train Acc',color='blue', linestyle='dashed')
    # plt.plot(x, get_max(normal_v_acc), label='TriHard Val Acc',color='green')
    # plt.plot(x,get_max(weighted_v_acc),label='TriWeighted Val Acc',color='blue')
    plt.plot(x, normal_g_ap, label='TriHard AP', color='red')
    plt.plot(x, normal_g_an, label='TriHard AN', color='red',linestyle='dashed')
    plt.plot(x, weighted_g_ap, label='TriWeighted AP', color='black')
    plt.plot(x, weighted_g_an, label='TriWeighted AN', color='black',linestyle='dashed')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.legend()
    plt.show()
    plt.savefig('fig.jpg')


def mutual_or_not():
    delimiters = mutual_training_info['train_acc_list'][0].delimiters \
    if len(training_info['train_acc'].delimiters)>len(mutual_training_info['train_acc_list'][0].delimiters) else training_info['train_acc'].delimiters
    delimiters.pop(0)
    delimiters = delimiters[:-3]
    # delimiters = np.array(delimiters)+1
    # delimiters = np.arange(3000)
    normal_acc=mean_(training_info['train_acc'].value_list,delimiters)
    model0_acc = mean_(mutual_training_info['train_acc_list'][0].value_list, delimiters)
    model1_acc = mean_(mutual_training_info['train_acc_list'][1].value_list, delimiters)

    normal_v_acc = mean_(training_info['val_acc'].value_list, delimiters)
    model0_v_acc = mean_(mutual_training_info['val_acc_list'][0].value_list, delimiters)
    model1_v_acc = mean_(mutual_training_info['val_acc_list'][1].value_list, delimiters)


    # normal_g_ap = mean_(training_info['train_g_ap'].value_list,delimiters)
    # normal_g_an = mean_(training_info['train_g_an'].value_list,delimiters)
    #
    # weighted_g_ap = mean_(weighted_training_info['train_g_ap'].value_list,delimiters)
    # weighted_g_an = mean_(weighted_training_info['train_g_an'].value_list,delimiters)

    # fig,(ax1,ax2)=plt.subplots(1,2)
    x = np.arange(len(normal_acc))+1
    # plt.title('Train Acc')
    # plt.plot(x,get_max(normal_acc),label='Single Model')
    # plt.plot(x, get_max(model0_acc), label='Mutual Learning Model 1',color='orange', linestyle='dashed')
    # plt.plot(x, get_max(model1_acc), label='Mutual Learning Model 2',color='orange')

    plt.title('Val Acc')
    plt.plot(x, get_max(normal_v_acc), label='Single Model')
    plt.plot(x, get_max(model0_v_acc), label='Mutual Learning Model 1', color='orange', linestyle='dashed')
    plt.plot(x, get_max(model1_v_acc), label='Mutual Learning Model 2', color='orange')

    # plt.plot(x,get_max(weighted_v_acc),label='TriWeighted Val Acc',color='blue')
    # plt.plot(x, normal_g_ap, label='TriHard AP', color='red')
    # plt.plot(x, normal_g_an, label='TriHard AN', color='red',linestyle='dashed')
    # plt.plot(x, weighted_g_ap, label='TriWeighted AP', color='black')
    # plt.plot(x, weighted_g_an, label='TriWeighted AN', color='black',linestyle='dashed')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig('fig.jpg')

if __name__ == '__main__':
    mutual_or_not()
    # dict_list=[mutual_training_info,training_info]
    # dict_list = clear(dict_list)
    # utils.dump_pickle(dict_list[0],osp.join(reid_cfg.training_info,'mutual_training_info.pkl'))
    # utils.dump_pickle(dict_list[1],osp.join(reid_cfg.training_info,'training_info.pkl'))



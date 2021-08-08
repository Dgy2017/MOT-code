import os

from gluoncv.data.transforms.block import RandomCrop
from mxnet import gpu,cpu
from mxnet.gluon.data.vision import transforms

from dataloader.dataloader import RandomSelectAug
from utils.utils import make_sure_dir


class AlignedReIdConfig(object):
    root = '/data/stu06/MCMTDataSet/code'
    use_class_loss = True
    local_loss_weight = 1
    global_loss_weight = 1
    class_loss_weight = 0.2

    use_fc = True
    classes = 1450

    input_width = 128
    input_height = 256

    # loss mode: one of {'weighted' or 'hardest'}
    local_loss_mode = 'hardest'
    global_loss_mode = 'hardest'

    # triplet loss margin
    global_margin = 0.5
    local_margin = 0.5

    # use gpu ?
    # ctx = cpu()
    ctx = gpu(1)

    gpu_num = 3
    ctx_list = [gpu(i + 1) for i in range(gpu_num)]
    batch_p = 8
    batch_k = 5
    log_interval = 20
    val_interval = 2
    random_seed = 970513
    learning_rate = 0.0002
    weight_decay = 0.0002
    val_split = 0.2
    epoch = 150
    decay_time = [20, 50, 80]
    decay_rate = 0.1
    momentum = 0.5
    clip_gradient = 5
    check_point = os.path.join(root, 'check_point/')
    make_sure_dir(check_point)
    patient = 30
    transforms = RandomSelectAug([
        transforms.RandomFlipLeftRight(),
        RandomCrop(size=(input_width, input_height), pad=10),
        # image.BrightnessJitterAug(0.5)
    ], 0.2)
    # transforms = None

    # test config
    image_dir = os.path.join(root, 'visual_result')
    make_sure_dir(image_dir)
    test_load_right = True
    training_info = os.path.join(root,'training_info')

    # mutual train

    mutual_model_ctx = [gpu(2), gpu(3)]
    mutual_model_number = len(mutual_model_ctx)
    mutual_model_pretrained = [False,False]
    load_parameters = True
    mutual_model_name = ['last_model1_0.8641.params', 'model_0.878_margin_0.3.params']

    # weighted
    weighted_models = ['model_0.8197_margin_1.params','last_model_0.8028.params']


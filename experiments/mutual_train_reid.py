import sys

sys.path.append('../')
from mxnet import gluon
from configs.alignedreid_config import AlignedReIdConfig
from configs.data_config import DataConfig
from dataloader.dataloader import CUHK03Loader
from models.resnet import resnet50_v1
from models.loss import GlobalLoss, LocalLoss
from mxnet import autograd
import utils.utils as utils
import time
from mxnet import nd
import os
import numpy as np
from experiments.metric import AvgMetric


def train_multi_model():
    logger = utils.get_logger()
    # utils.set_random_seed(AlignedReIdConfig.random_seed)

    # model.initialize(init=mx.init.Xavier(),ctx=AlignedReIdConfig.ctx)
    # model.initialize( ctx=AlignedReIdConfig.ctx)
    data_loader = CUHK03Loader(os.path.join(DataConfig.market1501_root, 'market1501.pkl'),
                               AlignedReIdConfig.batch_p,
                               AlignedReIdConfig.batch_k,
                               transforms=AlignedReIdConfig.transforms)
    model_list = [
        resnet50_v1(ctx=AlignedReIdConfig.mutual_model_ctx[id], pretrained=AlignedReIdConfig.mutual_model_pretrained[id],
                    use_fc=AlignedReIdConfig.use_fc,classes=data_loader.train_num)
        for id in range(AlignedReIdConfig.mutual_model_number)]
    if AlignedReIdConfig.load_parameters:
        for model,ctx,model_name in zip(model_list, AlignedReIdConfig.mutual_model_ctx, AlignedReIdConfig.mutual_model_name):
            model.load_parameters(os.path.join(AlignedReIdConfig.check_point,model_name),ctx=ctx)

    # for model in model_list:
    #     model.initialize()

    trainer_list = [gluon.Trainer(model_list[id].collect_params(), 'adam', {
        'learning_rate': AlignedReIdConfig.learning_rate,
        'wd': AlignedReIdConfig.weight_decay,
        'clip_gradient': AlignedReIdConfig.clip_gradient
    }) for id in range(AlignedReIdConfig.mutual_model_number)]

    g_loss_fun_list = [GlobalLoss(AlignedReIdConfig.global_margin, AlignedReIdConfig.global_loss_weight,
                                  AlignedReIdConfig.global_loss_mode, dist_mat=True) for id in
                       range(AlignedReIdConfig.mutual_model_number)]
    l_loss_fun_list = [LocalLoss(AlignedReIdConfig.local_margin, AlignedReIdConfig.local_loss_weight,
                                 AlignedReIdConfig.local_loss_mode, dist_mat=True) for id in
                       range(AlignedReIdConfig.mutual_model_number)]

    class_loss_list = [gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True) for id in
                       range(AlignedReIdConfig.mutual_model_number)]
    kl_loss_list = [gluon.loss.KLDivLoss(from_logits=False, axis=1) for id in
                    range(AlignedReIdConfig.mutual_model_number)]
    best_acc = 0
    patient = 0

    # train metric
    train_acc_list = [AvgMetric('T-Acc') for id in range(AlignedReIdConfig.mutual_model_number)]
    train_g_ap_list = [AvgMetric('T-g_ap') for id in range(AlignedReIdConfig.mutual_model_number)]
    train_g_an_list = [AvgMetric('T-g_an') for id in range(AlignedReIdConfig.mutual_model_number)]
    train_l_ap_list = [AvgMetric('T-l_ap') for id in range(AlignedReIdConfig.mutual_model_number)]
    train_l_an_list = [AvgMetric('T-l_an') for id in range(AlignedReIdConfig.mutual_model_number)]
    train_class_loss_list = [AvgMetric('C-Loss') for id in range(AlignedReIdConfig.mutual_model_number)]
    train_loss_list = [AvgMetric('T-Loss') for id in range(AlignedReIdConfig.mutual_model_number)]

    # val metric
    val_acc_list = [AvgMetric('V-Acc') for id in range(AlignedReIdConfig.mutual_model_number)]
    val_g_ap_list = [AvgMetric('V-g_ap') for id in range(AlignedReIdConfig.mutual_model_number)]
    val_g_an_list = [AvgMetric('V-g_an') for id in range(AlignedReIdConfig.mutual_model_number)]
    val_l_ap_list = [AvgMetric('V-l_ap') for id in range(AlignedReIdConfig.mutual_model_number)]
    val_l_an_list = [AvgMetric('V-l_an') for id in range(AlignedReIdConfig.mutual_model_number)]
    val_loss_list = [AvgMetric('V-Loss') for id in range(AlignedReIdConfig.mutual_model_number)]

    begin_time = time.time()
    logger.info(f"Training started at {time.strftime('%Y %m %d %H %M %S ',time.localtime(begin_time))}")
    for epoch_id in range(AlignedReIdConfig.epoch):

        alpha = np.clip(np.sqrt(0.0237*epoch_id+0.0163),0.2,0.7)

        if epoch_id in AlignedReIdConfig.decay_time:
            for id, trainer in enumerate(trainer_list):
                trainer.set_learning_rate(trainer.learning_rate * AlignedReIdConfig.decay_rate)
                logger.info(f'Epoch {epoch_id}: set model {id} lr to {trainer.learning_rate}')
        for train_acc, train_g_an, train_g_ap, train_l_an, train_l_ap, train_loss, train_class_loss \
                in zip(train_acc_list, train_g_an_list, train_g_ap_list, train_l_an_list, train_l_ap_list,
                       train_loss_list, train_class_loss_list):
            train_acc.reset()
            train_g_an.reset()
            train_g_ap.reset()
            train_l_an.reset()
            train_l_ap.reset()
            train_loss.reset()
            train_class_loss.reset()
        for i, (imgs, labels) in enumerate(data_loader.train_loader()):
            model_loss_list, g_dist_mat_list, l_dist_mat_list, g_ap_list, \
            g_an_list, l_ap_list, l_an_list, class_score_list, c_loss_list = [], [], [], [], [], [], [], [], []
            total_loss_list = []
            with autograd.record(True):
                for id, (model, g_loss_fun, l_loss_fun, class_loss) \
                        in enumerate(zip(model_list, g_loss_fun_list, l_loss_fun_list, class_loss_list)):
                    imgs = imgs.as_in_context(AlignedReIdConfig.mutual_model_ctx[id])
                    labels = labels.as_in_context(AlignedReIdConfig.mutual_model_ctx[id])

                    g_feature, l_feature, class_score = model(imgs)
                    g_loss, g_ap, g_an, g_dist_mat = g_loss_fun(g_feature, labels)
                    l_loss, l_ap, l_an, l_dist_mat = l_loss_fun(l_feature, labels)
                    c_loss = nd.mean(class_loss(class_score, labels))
                    single_model_loss = g_loss + l_loss + AlignedReIdConfig.class_loss_weight * c_loss
                    model_loss_list.append(single_model_loss)
                    c_loss_list.append(c_loss)

                    g_ap_list.append(g_ap)
                    g_an_list.append(g_an)
                    l_ap_list.append(l_ap)
                    l_an_list.append(l_an)

                    g_dist_mat_list.append(g_dist_mat)
                    l_dist_mat_list.append(l_dist_mat)
                    class_score_list.append(class_score)
                # todo
                # zero grad
                # with autograd.pause():
                #
                #     g_dist_mat_sum = nd.stack(*g_dist_mat_list).sum(axis=0)
                #     l_dist_mat_sum = nd.stack(*l_dist_mat_list).sum(axis=0)
                #     class_score_mean = nd.stack(*class_score_list).mean(axis=0)

                g_dist_mat_sum_numpy = np.stack([dist_mat.asnumpy() for dist_mat in g_dist_mat_list]).sum(axis=0)
                l_dist_mat_sum_numpy = np.stack([dist_mat.asnumpy() for dist_mat in l_dist_mat_list]).sum(axis=0)
                class_score_mean_numpy = np.stack([class_score.asnumpy() for class_score in class_score_list]).mean(
                    axis=0)
                class_score_mean_numpy = np.exp(class_score_mean_numpy)
                class_score_mean_numpy /= np.sum(class_score_mean_numpy, axis=1, keepdims=True)

                # todo
                for id, (loss, g_dist_mat, l_dist_mat, class_score, kl_loss) \
                        in enumerate(
                    zip(model_loss_list, g_dist_mat_list, l_dist_mat_list, class_score_list, kl_loss_list)):
                    # g_dist_mat_sum = g_dist_mat_sum.as_in_context(AlignedReIdConfig.mutual_model_ctx[id])
                    # l_dist_mat_sum = l_dist_mat_sum.as_in_context(AlignedReIdConfig.mutual_model_ctx[id])
                    # class_score_mean = class_score_mean.as_in_context(AlignedReIdConfig.mutual_model_ctx[id])
                    g_dist_mat_sum = nd.array(g_dist_mat_sum_numpy, ctx=AlignedReIdConfig.mutual_model_ctx[id])
                    l_dist_mat_sum = nd.array(l_dist_mat_sum_numpy, ctx=AlignedReIdConfig.mutual_model_ctx[id])
                    class_score_mean = nd.array(class_score_mean_numpy, ctx=AlignedReIdConfig.mutual_model_ctx[id])
                    n = AlignedReIdConfig.mutual_model_number
                    loss = (1 - alpha) * loss + alpha * (0.4 * nd.mean(nd.square((n * g_dist_mat - g_dist_mat_sum) / (n - 1) + 1e-8)) \
                           + 0.4 * nd.mean(nd.square((n * l_dist_mat - l_dist_mat_sum) / (n - 1) + 1e-8)) \
                           + 0.2 * nd.mean(kl_loss(class_score, class_score_mean)))
                    total_loss_list.append(loss)

            autograd.backward(total_loss_list)
            for trainer in trainer_list:
                trainer.step(batch_size=AlignedReIdConfig.batch_k * AlignedReIdConfig.batch_p,
                             ignore_stale_grad=False)

            if i % AlignedReIdConfig.log_interval == 0:
                logger.info(
                    f'---------------------------------------------------alpha {alpha:.4f}------------------------------------------------------------')
                for id, (train_acc, train_g_an, train_g_ap, train_l_an, train_l_ap, train_class_loss, train_loss, \
                         g_an, g_ap, l_an, l_ap, c_loss, loss) \
                        in enumerate(
                    zip(train_acc_list, train_g_an_list, train_g_ap_list, train_l_an_list, train_l_ap_list,
                        train_class_loss_list, train_loss_list,
                        g_an_list, g_ap_list, l_an_list, l_ap_list, c_loss_list, total_loss_list)):
                    acc = nd.sum(g_ap < g_an) / len(g_ap)
                    train_acc.update(acc.asscalar())
                    train_g_an.update(nd.mean(g_an).asscalar())
                    train_g_ap.update(nd.mean(g_ap).asscalar())
                    train_l_an.update(nd.mean(l_an).asscalar())
                    train_l_ap.update(nd.mean(l_ap).asscalar())
                    train_class_loss.update(nd.mean(c_loss))
                    train_loss.update(loss.asscalar())
                    logger.info(f'mode id {id} ' +
                                'Epoch:{} Sample:{} '.format(epoch_id, i) +
                                '{}:{:.4f} '.format(*train_loss.get()) +
                                '{}:{:.4f} '.format(*train_acc.get()) +
                                '{}:{:.4f} '.format(*train_g_ap.get()) +
                                '{}:{:.4f} '.format(*train_g_an.get()) +
                                '{}:{:.4f} '.format(*train_l_ap.get()) +
                                '{}:{:.4f} '.format(*train_l_an.get()) +
                                '{}:{:.4f}'.format(*train_class_loss.get())
                                )
                # logger.info(f"Epoch:{epoch_id} Sample:{i} Train-Loss:{loss.asscalar():.4f} Train-Acc:{train_acc.get() :.4f}")

        if epoch_id % AlignedReIdConfig.val_interval == 0:
            logger.info(
                '********************************************************************************************************************************')
            for val_acc, val_g_an, val_g_ap, val_l_an, val_l_ap, val_loss \
                    in zip(val_acc_list, val_g_an_list, val_g_ap_list, val_l_an_list, val_l_ap_list, val_loss_list):
                val_acc.reset()
                val_g_an.reset()
                val_g_ap.reset()
                val_l_an.reset()
                val_l_ap.reset()
                val_loss.reset()
            logger.info('validating...')
            for imgs, labels in data_loader.val_loader(12, 4):
                for id, (model, g_loss_fun, l_loss_fun, class_loss,
                         val_acc, val_g_an, val_g_ap, val_l_an, val_l_ap, val_loss) \
                        in enumerate(zip(model_list, g_loss_fun_list, l_loss_fun_list, class_loss_list,
                                         val_acc_list, val_g_an_list, val_g_ap_list, val_l_an_list, val_l_ap_list,
                                         val_loss_list)):
                    imgs = imgs.as_in_context(AlignedReIdConfig.mutual_model_ctx[id])
                    labels = labels.as_in_context(AlignedReIdConfig.mutual_model_ctx[id])
                    g_feature, l_feature, _ = model(imgs)
                    g_loss, g_ap, g_an, _ = g_loss_fun(g_feature, labels)
                    l_loss, l_ap, l_an, _ = l_loss_fun(l_feature, labels)
                    loss = g_loss + l_loss
                    acc = nd.sum(g_ap < g_an) / len(g_ap)
                    val_acc.update(acc)
                    val_g_an.update(nd.mean(g_an).asscalar())
                    val_g_ap.update(nd.mean(g_ap).asscalar())
                    val_l_an.update(nd.mean(l_an).asscalar())
                    val_l_ap.update(nd.mean(l_ap).asscalar())
                    val_loss.update(loss.asscalar())

            elapse = time.time() - begin_time
            time_batch = (elapse) / (
                    (epoch_id + 1) * len(data_loader) * AlignedReIdConfig.batch_p * AlignedReIdConfig.batch_k)
            total = (elapse / (epoch_id + 1)) * AlignedReIdConfig.epoch
            for id, (val_acc, val_g_an, val_g_ap, val_l_an, val_l_ap, val_loss) \
                    in enumerate(
                zip(val_acc_list, val_g_an_list, val_g_ap_list, val_l_an_list, val_l_ap_list, val_loss_list)):
                logger.info(f'model id {id} ' +
                            f"Epoch:{epoch_id} " +
                            f'Speed:{time_batch:.4}s/batch ' +
                            '{}:{:.4f} '.format(*val_loss.get()) +
                            '{}:{:.4f} '.format(*val_acc.get()) +
                            '{}:{:.4f} '.format(*val_g_ap.get()) +
                            '{}:{:.4f} '.format(*val_g_an.get()) +
                            '{}:{:.4f} '.format(*val_l_ap.get()) +
                            '{}:{:.4f} '.format(*val_l_an.get()) +
                            f'Elapse:{time.time()-begin_time:.4}s/{total/3600:.4}h')
                acc = val_acc.get()[1]
                if acc > best_acc:
                    best_acc = acc
                    patient = 0
                    model_list[id].save_parameters(
                        os.path.join(AlignedReIdConfig.check_point, f'model{id}_{acc:.4}.params'))
                else:
                    patient += 1
            if patient > AlignedReIdConfig.patient:
                for id, model in enumerate(model_list):
                    model.save_parameters(
                        os.path.join(AlignedReIdConfig.check_point, f'last_model{id}_{acc:.4}.params'))
                    logger.info(f'Early stoping at {epoch_id+1}')
                break

        training_info = dict(
            epoch_id=epoch_id,
            train_acc_list=train_acc_list,
            train_g_ap_list=train_g_ap_list,
            train_g_an_list=train_g_an_list,
            train_l_ap_list=train_l_ap_list,
            train_l_an_list=train_l_an_list,
            train_class_loss_list=train_class_loss_list,
            train_loss_list=train_loss_list,
            val_acc_list=val_acc_list,
            val_g_ap_list=val_g_ap_list,
            val_g_an_list=val_g_an_list,
            val_l_ap_list=val_l_ap_list,
            val_l_an_list=val_l_an_list,
            val_loss_list=val_loss_list
        )
        utils.dump_pickle(training_info, os.path.join(AlignedReIdConfig.training_info, 'pretrained_mutual_training_info.pkl'))


    logger.info(f"Training finished at {time.strftime('%Y %m %d %H %M %S ',time.localtime(time.time()))}")


if __name__ == '__main__':
    train_multi_model()

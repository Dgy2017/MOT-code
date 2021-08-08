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
from experiments.metric import AvgMetric


def train_single_modle():
    logger = utils.get_logger()
    utils.set_random_seed(AlignedReIdConfig.random_seed)

    # model.initialize(init=mx.init.Xavier(),ctx=AlignedReIdConfig.ctx)
    # model.initialize( ctx=AlignedReIdConfig.ctx)
    data_loader = CUHK03Loader(os.path.join(DataConfig.market1501_root, 'market1501.pkl'),
                               AlignedReIdConfig.batch_p,
                               AlignedReIdConfig.batch_k,
                               transforms=AlignedReIdConfig.transforms)
    model = resnet50_v1(ctx=AlignedReIdConfig.ctx, pretrained=True, use_fc=AlignedReIdConfig.use_fc,
                        classes=data_loader.train_num)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {
        'learning_rate': AlignedReIdConfig.learning_rate,
        'wd': AlignedReIdConfig.weight_decay,
        'clip_gradient': AlignedReIdConfig.clip_gradient
    })
    g_loss_fun = GlobalLoss(1, AlignedReIdConfig.global_loss_weight,
                            AlignedReIdConfig.global_loss_mode,dist_mat=True)
    l_loss_fun = LocalLoss(1, AlignedReIdConfig.local_loss_weight,
                           AlignedReIdConfig.local_loss_mode)
    class_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)
    best_acc = 0
    patient = 0

    # train metric
    train_acc = AvgMetric('T-Acc')
    train_g_ap = AvgMetric('T-g_ap')
    train_g_an = AvgMetric('T-g_an')
    train_l_ap = AvgMetric('T-l_ap')
    train_l_an = AvgMetric('T-l_an')
    train_class_loss = AvgMetric('C-Loss')
    train_loss = AvgMetric('T-Loss')

    # val metric
    val_acc = AvgMetric('V-Acc')
    val_g_ap = AvgMetric('V-g_ap')
    val_g_an = AvgMetric('V-g_an')
    val_l_ap = AvgMetric('V-l_ap')
    val_l_an = AvgMetric('V-l_an')
    val_loss = AvgMetric('V-Loss')

    begin_time = time.time()
    logger.info(f"Training started at {time.strftime('%Y %m %d %H %M %S ',time.localtime(begin_time))}")
    for epoch_id in range(AlignedReIdConfig.epoch):
        if epoch_id in AlignedReIdConfig.decay_time:
            trainer.set_learning_rate(trainer.learning_rate * AlignedReIdConfig.decay_rate)
            logger.info(f'Epoch {epoch_id}: set lr to {trainer.learning_rate}')
        train_acc.reset()
        train_g_an.reset()
        train_g_ap.reset()
        train_l_an.reset()
        train_l_ap.reset()
        train_loss.reset()
        train_class_loss.reset()
        for i, (imgs, labels) in enumerate(data_loader.train_loader()):
            with autograd.record(True):

                imgs = imgs.as_in_context(AlignedReIdConfig.ctx)
                labels = labels.as_in_context(AlignedReIdConfig.ctx)

                g_feature, l_feature, class_score = model(imgs)
                g_loss, g_ap, g_an,dis_mat = g_loss_fun(g_feature, labels)
                l_loss, l_ap, l_an = l_loss_fun(l_feature, labels)
                c_loss = nd.mean(class_loss(class_score, labels))
                loss = g_loss + l_loss + AlignedReIdConfig.class_loss_weight * c_loss
            loss.backward()
            trainer.step(batch_size=AlignedReIdConfig.batch_k * AlignedReIdConfig.batch_p, ignore_stale_grad=True)
            if i % AlignedReIdConfig.log_interval == 0:
                # dis_mat[nd.arange(len(dis_mat), ctx=dis_mat.context), nd.arange(len(dis_mat), ctx=dis_mat.context)] = 1e10
                # index = nd.argmin(dis_mat,axis=1)
                # pred = labels[index]
                #
                # acc = nd.sum(pred==labels) / len(labels)
                acc = nd.sum(g_ap < g_an) / len(g_ap)
                train_acc.update(acc)
                train_g_an.update(nd.mean(g_an).asscalar())
                train_g_ap.update(nd.mean(g_ap).asscalar())
                train_l_an.update(nd.mean(l_an).asscalar())
                train_l_ap.update(nd.mean(l_ap).asscalar())
                train_class_loss.update(nd.mean(c_loss))
                train_loss.update(loss.asscalar())
                logger.info(
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
            val_acc.reset()
            val_g_an.reset()
            val_g_ap.reset()
            val_l_an.reset()
            val_l_ap.reset()
            val_loss.reset()
            logger.info('validating...')
            for imgs, labels in data_loader.val_loader(12, 4):
                imgs = imgs.as_in_context(AlignedReIdConfig.ctx)
                labels = labels.as_in_context(AlignedReIdConfig.ctx)
                g_feature, l_feature, _ = model(imgs)
                g_loss, g_ap, g_an,dis_mat = g_loss_fun(g_feature, labels)
                l_loss, l_ap, l_an = l_loss_fun(l_feature, labels)
                loss = g_loss + l_loss
                # dis_mat[
                #     nd.arange(len(dis_mat), ctx=dis_mat.context), nd.arange(len(dis_mat), ctx=dis_mat.context)] = 1e10
                # index = nd.argmin(dis_mat, axis=1)
                # pred = labels[index]
                # acc = nd.sum(pred == labels) / len(labels)
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
            logger.info(f"Epoch:{epoch_id} "
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
                model.save_parameters(os.path.join(AlignedReIdConfig.check_point, f'model_{acc:.4}.params'))
            else:
                patient += 1
            if patient > AlignedReIdConfig.patient:
                model.save_parameters(os.path.join(AlignedReIdConfig.check_point, f'last_model_{acc:.4}.params'))
                logger.info(f'Early stoping at {epoch_id+1}')
                break

        training_info = dict(
            epoch_id=epoch_id,
            train_acc=train_acc,
            train_g_ap=train_g_ap,
            train_g_an=train_g_an,
            train_l_ap=train_l_ap,
            train_l_an=train_l_an,
            train_class_loss=train_class_loss,
            train_loss=train_loss,
            val_acc=val_acc,
            val_g_ap=val_g_ap,
            val_g_an=val_g_an,
            val_l_ap=val_l_ap,
            val_l_an=val_l_an,
            val_loss=val_loss
        )
        utils.dump_pickle(training_info, os.path.join(AlignedReIdConfig.training_info, 'training_info.pkl'))

    logger.info(f"Training finished at {time.strftime('%Y %m %d %H %M %S ',time.localtime(time.time()))}")


if __name__ == '__main__':
    train_single_modle()

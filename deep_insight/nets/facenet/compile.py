#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/21 16:28
:File   :compile.py
:content:
  
"""

import unittest

import tensorflow as tf

from deep_insight import *
from deep_insight.base import *
from deep_insight.nets.facenet.model import FaceNetSoftMax


class FaceNetSoftMaxCompile(Base):
    """
    参数：
        prelogits_norm_p: Norm to use for prelogits norm loss.
        prelogits_norm_loss_factor: Loss based on the norm of the activations in the prelogits layer.
        center_loss_alfa: Center update rate for center loss.
        center_loss_factor: Center loss factor.
        learning_rate_decay_epochs: Number of epochs between learning rate decay.
        learning_rate_decay_factor: Learning rate decay factor.
        optimizer: The optimization algorithm to use
        moving_average_decay: Exponential decay for tracking of training parameters.
        log_histograms: Enables logging of weight/bias histograms in tensorboard.
    """

    def __init__(self, prelogits_norm_p='1.0',
                 prelogits_norm_loss_factor='0.0',
                 center_loss_alfa='0.95',
                 center_loss_factor='0.0',
                 learning_rate_decay_epochs='100',
                 learning_rate_decay_factor='1.0',
                 optimizer='ADAGRAD',
                 moving_average_decay='0.9999',
                 log_histograms='false'):
        super(FaceNetSoftMaxCompile, self).__init__()
        self.p('prelogits_norm_p', prelogits_norm_p)
        self.p('prelogits_norm_loss_factor', prelogits_norm_loss_factor)
        self.p('center_loss_alfa', center_loss_alfa)
        self.p('center_loss_factor', center_loss_factor)
        self.p('learning_rate_decay_epochs', learning_rate_decay_epochs)
        self.p('learning_rate_decay_factor', learning_rate_decay_factor)
        self.p('optimizer', optimizer)
        self.p('moving_average_decay', moving_average_decay)
        self.p('log_histograms', log_histograms)

    def run(self):
        param = self.params

        from tfos.nets.facenet.compile import FaceNetSoftMaxCompile
        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool

        OPTIMIZERS = ['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM']

        # param = json.loads('<#zzjzParam#>')
        input_prev_layers = param.get('input_prev_layers')
        prelogits_norm_p = param.get('prelogits_norm_p', 1.0)
        prelogits_norm_loss_factor = param.get('prelogits_norm_loss_factor', 0.0)
        center_loss_alfa = param.get('center_loss_alfa', 0.95)
        center_loss_factor = param.get('center_loss_factor', 0.0)
        learning_rate_decay_epochs = param.get('learning_rate_decay_epochs', 100)
        learning_rate_decay_factor = param.get('learning_rate_decay_factor', 1.0)
        optimizer = param.get('optimizer', OPTIMIZERS[0])
        moving_average_decay = param.get('moving_average_decay', 0.9999)
        log_histograms = param.get('log_histograms', BOOLEAN[1])

        model_rdd = inputRDD(input_prev_layers)
        assert model_rdd, "cannot get model config rdd from previous model layer!"

        kwargs = dict()
        if prelogits_norm_p:
            prelogits_norm_p = float(prelogits_norm_p)
            assert 0 <= prelogits_norm_p <= 1, ""
            kwargs['prelogits_norm_p'] = prelogits_norm_p
        if prelogits_norm_loss_factor:
            prelogits_norm_loss_factor = float(prelogits_norm_loss_factor)
            assert 0 <= prelogits_norm_loss_factor <= 1, ""
            kwargs['prelogits_norm_loss_factor'] = prelogits_norm_loss_factor
        if center_loss_alfa:
            center_loss_alfa = float(center_loss_alfa)
            assert 0 <= center_loss_alfa <= 1, ""
            kwargs['center_loss_alfa'] = center_loss_alfa
        if center_loss_factor:
            center_loss_factor = float(center_loss_factor)
            assert 0 <= center_loss_factor <= 1, ""
            kwargs['center_loss_factor'] = center_loss_factor
        if learning_rate_decay_epochs:
            learning_rate_decay_epochs = int(learning_rate_decay_epochs)
            assert learning_rate_decay_epochs > 0, ""
            kwargs['learning_rate_decay_epochs'] = learning_rate_decay_epochs
        if learning_rate_decay_factor:
            learning_rate_decay_factor = float(learning_rate_decay_factor)
            assert 0 <= learning_rate_decay_factor <= 1, ""
            kwargs['learning_rate_decay_factor'] = learning_rate_decay_factor
        if optimizer:
            kwargs['optimizer'] = optimizer
        if moving_average_decay:
            moving_average_decay = float(moving_average_decay)
            assert 0 <= moving_average_decay <= 1, ""
            kwargs['moving_average_decay'] = moving_average_decay
        if log_histograms:
            log_histograms = convert_bool(log_histograms)
            kwargs['log_histograms'] = log_histograms

        output_df = FaceNetSoftMaxCompile(model_rdd, **kwargs).compile().serialize(sqlc)
        output_df.show()
        outputRDD('<#zzjzRddName#>_Masking', output_df)


class TestFaceNetSoftMax(unittest.TestCase):
    def setUp(self) -> None:
        self.is_local = True
        self.input_dir = os.path.join(self.path, "data/data/lfw")
        self.output_dir = os.path.join(self.path, "data/data/lfw_160_30")

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    def test_facenet_softmax_compile(self):
        FaceNetSoftMax(image_size='160', n_classes='5749').run()
        tf.reset_default_graph()
        FaceNetSoftMaxCompile().run()


if __name__ == '__main__':
    unittest.main()

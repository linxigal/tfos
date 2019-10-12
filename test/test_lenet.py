#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/19 11:28
:File   :test_lenet.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *
from deep_insight.compile import Compile
from deep_insight.data.mnist import Mnist
from deep_insight.layers.convolution import Convolution2D
from deep_insight.layers.core import Dense, Flatten
from deep_insight.layers.input import InputLayer
from deep_insight.layers.pooling import MaxPool2D
from deep_insight.model.model import TrainModel, EvaluateModel, PredictModel


class TestLeNet(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = True
        self.mnist_dir = os.path.join(self.path, 'data/data/mnist')
        self.model_dir = os.path.join(self.path, 'data/model/LeNet')

    def tearDown(self) -> None:
        reset()

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    @staticmethod
    def build_model():
        m = MODEL_BRANCH
        # build model
        InputLayer('28,28,1').b(m).run()
        Convolution2D(filters='20', kernel_size='5,5', activation='relu', padding='valid').run()
        MaxPool2D(pool_size='2,2', strides='2,2').run()
        Convolution2D(filters='50', kernel_size='5,5', activation='relu', padding='valid').run()
        MaxPool2D(pool_size='2,2', strides='2,2').run()
        Flatten().run()
        Dense('120', activation='relu').run()
        Dense('10', activation='softmax').run()
        # compile model
        Compile('categorical_crossentropy', 'sgd', ['accuracy']).run()
        # show network struct
        SummaryLayer(m).run()

    @unittest.skip("")
    def test_lenet_train(self):
        # load train data
        Mnist(self.mnist_dir, mode='train', is_conv='true').b(DATA_BRANCH).run()
        self.build_model()

        # model train
        TrainModel(input_prev_layers=MODEL_BRANCH,
                   input_rdd_name=DATA_BRANCH,
                   cluster_size=2,
                   num_ps=1,
                   batch_size=32,
                   epochs=2,
                   model_dir=self.model_dir,
                   go_on='false').run()

    @unittest.skip("")
    def test_lenet_evaluate(self):
        # load train data
        Mnist(self.mnist_dir, mode='test', is_conv='true').b(DATA_BRANCH).run()
        # model train
        EvaluateModel(input_prev_layers=MODEL_BRANCH,
                      input_rdd_name=DATA_BRANCH,
                      cluster_size=2,
                      num_ps=1,
                      steps=0,
                      model_dir=self.model_dir).run()

    @unittest.skip("")
    def test_lenet_predict(self):
        # load train data
        Mnist(self.mnist_dir, mode='test', is_conv='true').b(DATA_BRANCH).run()
        # model predict
        PredictModel(input_prev_layers=MODEL_BRANCH,
                     input_rdd_name=DATA_BRANCH,
                     cluster_size=2,
                     num_ps=1,
                     steps=10,
                     model_dir=self.model_dir,
                     output_prob='true').run()


if __name__ == '__main__':
    unittest.main()

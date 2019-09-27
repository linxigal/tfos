#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/26 17:26
:File   :test_model_compile.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *
from deep_insight.compile import CompileAdvanced
from deep_insight.data.mnist import Mnist
from deep_insight.layers.core import Dropout, Dense
from deep_insight.layers.input import InputLayer
from deep_insight.layers.optimizer import SGD
from deep_insight.model.model import TrainModel


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = True
        self.mnist_dir = os.path.join(self.path, 'data/data/mnist')
        self.model_dir = os.path.join(self.path, 'data/model/mnist_mlp')

    def tearDown(self) -> None:
        reset()

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    @staticmethod
    def build_model():
        n = -1
        m = MODEL_BRANCH
        SGD().b(n).run()
        # build model
        InputLayer('784').b(m).run()
        Dense('512', activation='relu').run()
        # Dense('512', activation='relu', input_shape='784').b(m).run()
        Dropout('0.2').run()
        Dense('512', activation='relu').run()
        Dropout('0.2').run()
        Dense('10', activation='softmax').run()
        # compile model
        CompileAdvanced(n, 'categorical_crossentropy', ['accuracy']).run()
        # show network struct
        SummaryLayer(m).run()

    # @unittest.skip("")
    def test_train_model(self):
        # load train data
        Mnist(self.mnist_dir, mode='train').b(DATA_BRANCH).run()
        self.build_model()
        # model train
        TrainModel(input_prev_layers=MODEL_BRANCH,
                   input_rdd_name=DATA_BRANCH,
                   cluster_size=2,
                   num_ps=1,
                   batch_size=32,
                   epochs=2,
                   model_dir=self.model_dir).run()


if __name__ == '__main__':
    unittest.main()

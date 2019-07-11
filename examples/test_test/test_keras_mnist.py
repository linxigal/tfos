#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:24
:File   : test_keras_mnist.py
"""

import os

from examples import ROOT_PATH
from examples.test_layer import TestDense, TestDropout
from examples.test_data.test_read_mnist import TestReadMnist
from examples.test_model.test_train import TestTrainModel
from examples.test_model.test_inference import TestInferenceModel
from examples.test_optimizer.test_optimizer import TestOptimizer
from examples.base import lrn

# load data
output_data_name = "<#zzjzRddName#>_data"
# build model
TestDense(lrn(), 512, activation='relu', input_dim=784).run()
TestDropout(lrn(), 0.2).run()
TestDense(lrn(), 512, activation='relu').run()
TestDropout(lrn(), 0.2).run()
TestDense(lrn(), 10, activation='softmax').run()

# compile model
TestOptimizer(lrn(), 'categorical_crossentropy', 'rmsprop', ['accuracy']).run()

# train model
model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")


def train_model():
    input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/train"
    # input_path = "/Users/wjl/github/tfos/output_data/mnist/tfr/train"
    TestReadMnist(input_path, 'tfr').run()
    TestTrainModel(output_data_name, lrn(),
                   cluster_size=2,
                   num_ps=1,
                   batch_size=1000,
                   epochs=1,
                   model_dir=model_dir).run()


def inference_model():
    input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/test"
    # input_path = "/Users/wjl/github/tfos/output_data/mnist/tfr/test"
    TestReadMnist(input_path, 'tfr').run()
    TestInferenceModel(output_data_name, lrn(),
                       cluster_size=2,
                       num_ps=1,
                       model_dir=model_dir).run()


if __name__ == "__main__":
    train_model()
    # inference_model()

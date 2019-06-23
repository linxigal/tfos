#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:24
:File   : test_keras_mnist.py
"""

import os

from examples import ROOT_PATH
from examples.test_layer.test_dense import TestDense
from examples.test_layer.test_drop import TestDrop
from examples.test_data.test_read_mnist import TestReadMnist
from examples.test_model.test_train import TestTrainModel
from examples.test_model.test_inference import TestInferenceModel
from examples.test_optimizer.test_optimizer import TestOptimizer

# load data
output_data_name = "<#zzjzRddName#>_mnist_tfr"
# build model
output_model_name = "<#zzjzRddName#>_model"
TestDense(output_model_name, 512, activation='relu', input_dim=784).run()
TestDrop(output_model_name, 0.2)
TestDense(output_model_name, 512, activation='relu').run()
TestDrop(output_model_name, 0.2)
TestDense(output_model_name, 10, activation='softmax').run()

# compile model
output_model_name = "<#zzjzRddName#>_model"
TestOptimizer(output_model_name, 'categorical_crossentropy', 'rmsprop', ['accuracy']).run()

# train model
model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")


def train_model():
    # input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/train"
    input_path = "/Users/wjl/github/tfos/output_data/mnist/tfr/train"
    TestReadMnist(output_data_name, input_path, 'tfr').run()
    TestTrainModel(output_data_name, output_model_name,
                   cluster_size=2,
                   num_ps=1,
                   batch_size=10,
                   epochs=10,
                   model_dir=model_dir).run()


def inference_model():
    # input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/test"
    input_path = "/Users/wjl/github/tfos/output_data/mnist/tfr/test"
    TestReadMnist(output_data_name, input_path, 'tfr').run()
    TestInferenceModel(output_data_name, output_model_name,
                       cluster_size=2,
                       num_ps=1,
                       batch_size=10,
                       model_dir=model_dir).run()


if __name__ == "__main__":
    # train_model()
    inference_model()

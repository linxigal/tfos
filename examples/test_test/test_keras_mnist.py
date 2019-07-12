#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/17 15:24
:File   : test_keras_mnist.py
:Content:
    example train path:
        # input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/train"
        # input_path = "/Users/wjl/github/tfos/output_data/mnist/tfr/train"
    example inference path:
        # input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/test"
        # input_path = "/Users/wjl/github/tfos/output_data/mnist/tfr/test"
    execute model:
        spark-submit    --master ${MASTER} \
                        --jars /home/wjl/github/TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
                        --num-executors 16 \
                        --executor-cores 16\
                        --executor-memory 4G \
                        examples/test_test/test_keras_mnist.py \
                        --mode train \
                        --format tfr \
                        --cluster_size 10 \
                        --epochs 20 \
                        --input_path /home/wjl/github/tfos/output_data/mnist/tfr/train
                        --model_dir /home/wjl/github/tfos/output_data/model_dir
"""

import argparse
import os

from examples import ROOT_PATH
from examples.base import lrn
from examples.test_data.test_read_mnist import TestReadMnist
from examples.test_layer import TestDense, TestDropout
from examples.test_model.test_inference import TestInferenceModel
from examples.test_model.test_train import TestTrainModel
from examples.test_optimizer.test_optimizer import TestOptimizer

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


def train_model(input_path, format, cluster_size, num_ps, batch_size, epochs, model_dir, **kwargs):
    TestReadMnist(input_path, format).run()
    TestTrainModel(output_data_name, lrn(),
                   cluster_size=cluster_size,
                   num_ps=num_ps,
                   batch_size=batch_size,
                   epochs=epochs,
                   model_dir=model_dir).run()


def inference_model(input_path, format, cluster_size, num_ps, model_dir, **kwargs):
    TestReadMnist(input_path, format).run()
    TestInferenceModel(output_data_name, lrn(),
                       cluster_size=cluster_size,
                       num_ps=num_ps,
                       model_dir=model_dir).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")
    parser.add_argument("--mode", help="execute: (train|inference)", choices=['train', 'inference'], default='train')
    parser.add_argument("--format", help="example format: (csv|tfr)", choices=["csv", "tfr"], default="tfr")
    parser.add_argument("--input_path", help="HDFS path to MNIST data")
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=2)
    parser.add_argument("--num_ps", help="number of parameter server", type=int, default=1)
    parser.add_argument("--batch_size", help="number of records per batch", type=int, default=1000)
    parser.add_argument("--epochs", help="number of epochs", type=int, default=1)
    parser.add_argument("--model_dir", help="HDFS path to save/load model during train/inference", default=model_dir)
    args = parser.parse_args()

    if not args.input_path or not args.model_dir:
        raise ValueError("Parameter 'input_path' or 'model_dir' must not be empty!")

    if args.mode == 'train':
        train_model(**args.__dict__)
    elif args.mode == 'inference':
        inference_model(**args.__dict__)
    else:
        parser.print_help()

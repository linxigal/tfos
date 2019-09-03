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
    standalone schema execute:
        spark-submit    --master ${MASTER} \
                        --jars /home/wjl/github/TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
                        --num-executors 3 \
                        --executor-cores 1\
                        --executor-memory 8G \
                        deep_insight/test/test_keras_mnist.py \
                        --mode train \
                        --format tfr \
                        --cluster_size 10 \
                        --epochs 20 \
                        --input_path /home/wjl/github/tfos/data/mnist/tfr/train \
                        --model_dir /home/wjl/github/tfos/data/model_dir

    yarn schema execute:
        spark-submit    --master yarn \
                        --queue ${QUEUE} \
                        --conf spark.dynamicAllocation.enabled=false \
                        --conf spark.yarn.maxAppAttempts=1 \
                        --conf spark.executorEnv.LD_LIBRARY_PATH=$LIB_JVM:$LIB_HDFS  \
                        --jars /home/wjl/github/TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar \
                        --num-executors 3 \
                        --executor-cores 1 \
                        --executor-memory 4G \
                        deep_insight/test/test_keras_mnist.py \
                        --mode train \
                        --format tfr \
                        --cluster_size 3 \
                        --epochs 1 \
                        --input_path hdfs://t-master:8020/data/mnist/tfr/test \
                        --model_dir hdfs://t-master:8020/data/model/mnist_mlp
"""

import argparse
import os

from deep_insight import ROOT_PATH
from deep_insight.base import lrn
from deep_insight.data.read_mnist import ReadMnist
from deep_insight.layers.core import Dense, Dropout
from deep_insight.model.inference import InferenceModel
from deep_insight.model.train import TrainModel
from deep_insight.optimizers.optimizer import Optimizer

# load data
output_data_name = "<#zzjzRddName#>_data"
# build model
Dense(lrn(), '512', activation='relu', input_shape='784').run()
Dropout(lrn(), '0.2').run()
Dense(lrn(), '512', activation='relu').run()
Dropout(lrn(), '0.2').run()
Dense(lrn(), '10', activation='softmax').run()

# compile model
Optimizer(lrn(), 'categorical_crossentropy', 'rmsprop', ['accuracy']).run()


def train_model(input_path, format, cluster_size, num_ps, batch_size, epochs, model_dir, **kwargs):
    ReadMnist(input_path, format).run()
    TrainModel(output_data_name, lrn(),
               cluster_size=cluster_size,
               num_ps=num_ps,
               batch_size=batch_size,
               epochs=epochs,
               model_dir=model_dir).run()


def inference_model(input_path, format, cluster_size, num_ps, model_dir, **kwargs):
    ReadMnist(input_path, format).run()
    InferenceModel(output_data_name, lrn(),
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
        train_model(**vars(args))
    elif args.mode == 'inference':
        inference_model(**vars(args))
    else:
        parser.print_help()

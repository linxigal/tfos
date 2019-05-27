# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 22:46
:File    : __init__.py.py
"""

import os
from collections import namedtuple
from os.path import join
from pyspark import SparkConf, SparkContext
import tensorflow as tf

CURRENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))
GITHUB = os.path.dirname(ROOT_PATH)
OUTPUT_DATA = os.path.join(ROOT_PATH, 'output_data')

if not os.path.exists(OUTPUT_DATA):
    os.makedirs(OUTPUT_DATA)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_path", join(ROOT_PATH, ""), "输入文件路径")
tf.app.flags.DEFINE_string("model_path", join(ROOT_PATH, "output_data"), "模型保存路径")
tf.app.flags.DEFINE_string("mode", "train", "train|inference|predict")
tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
tf.app.flags.DEFINE_integer("cluster_size", 2, "num executor")
tf.app.flags.DEFINE_integer("epochs", 1, "num epochs")
tf.app.flags.DEFINE_integer("num_ps", 1, "num ps")
tf.app.flags.DEFINE_integer("steps", 1000, "steps")
tf.app.flags.DEFINE_integer("rdma", 0, "rdma")
tf.app.flags.DEFINE_integer("tensorboard", 1, "tensorboard")
tf.app.flags.DEFINE_intege("readers", 10, "number of reader/enqueue threads per worker")
tf.app.flags.DEFINE_intege("shuffle_size", 1000,  help="size of shuffle buffer")
sc = SparkContext(conf=SparkConf().setAppName('logistic_regression')
                  .setMaster('spark://192.168.209.128:7077')
                  # .setMaster('local[*]')
                  # .set('spark.py-files ', '/home/wjl/github/tfos')
                  .set("spark.jars", join(GITHUB, "TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar")))
ARGS = namedtuple("args", ['batch_size', 'mode', 'steps', 'model_path', 'rdma'])
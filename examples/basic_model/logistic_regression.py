#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
:Author     :weijinlong
:Time: 2019/5/22 17:18
:File       : logicstic_regression.py
"""

from collections import namedtuple
from os.path import join

from pyspark import SparkConf, SparkContext

from examples.basic_model import GITHUB, ROOT_PATH, TENSORBOARD
from tfos.TFOS import TFOSRdd
from tfos.utils.op_example import *
from tfos.contrib.layers.regression import logistic_regression_layer

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("output_path", join(ROOT_PATH, "output_data/iris.tfr"), "输出文件路径")
tf.app.flags.DEFINE_string("input_path", join(ROOT_PATH, "output_data/iris.tfr"), "输入文件路径")
tf.app.flags.DEFINE_string("model_path", join(ROOT_PATH, "output_data"), "训练模型保存路径")
tf.app.flags.DEFINE_string("mode", "train", "train|inference")
tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
tf.app.flags.DEFINE_integer("cluster_size", 2, "num executor")
tf.app.flags.DEFINE_integer("epochs", 1, "num epochs")
tf.app.flags.DEFINE_integer("num_ps", 1, "num ps")
tf.app.flags.DEFINE_integer("steps", 1000, "steps")
tf.app.flags.DEFINE_integer("rdma", 0, "rdma")
tf.app.flags.DEFINE_integer("tensorboard", 1, "tensorboard")
# tf.app.flags.DEFINE_string("tb_path", TENSORBOARD, "tensorboard log file path")
sc = SparkContext(conf=SparkConf().setAppName('data_trans')
                  # .setMaster('spark://192.168.209.128:7077')
                  # .setMaster('spark://localhost:7077')
                  # .setMaster('local[*]')
                  .set("spark.jars", join(GITHUB, "TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar")))
ARGS = namedtuple("args", ['batch_size', 'mode', 'steps', 'model_path', 'rdma', 'tb_path'])


# sys.path.append('/home/wjl/github/tfos')


def main(unused_args):
    rdd = input_example(sc, FLAGS.input_path)
    tfos = TFOSRdd(sc, 'logistic_regression', **FLAGS.flag_values_dict())
    tfos.init_tfos(logistic_regression_layer, rdd)
    tfos.train()


if __name__ == "__main__":
    tf.app.run()

# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 20:15
:File    : logistic_regression_main.py
"""

import sys
sys.path.append('/home/wjl/github/tfos')
import logging
from os.path import join
from collections import namedtuple

from pyspark import SparkConf, SparkContext
from sklearn.datasets import load_iris
from tensorflowonspark import TFCluster
from examples.basic_model import GITHUB, ROOT_PATH, TENSORBOARD, logistic_regression_dist
from tfos.utils.op_example import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("output_path", join(ROOT_PATH, "output_data/iris.tfr"), "输出文件路径")
tf.app.flags.DEFINE_string("input_path", join(ROOT_PATH, "output_data/iris.tfr"), "输入文件路径")
tf.app.flags.DEFINE_string("model_path", join(ROOT_PATH, "output_data/logistic_regression_model"), "训练模型保存路径")
tf.app.flags.DEFINE_string("mode", "train", "train|inference")
tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
tf.app.flags.DEFINE_integer("cluster_size", 2, "num executor")
tf.app.flags.DEFINE_integer("epochs", 1, "num epochs")
tf.app.flags.DEFINE_integer("num_ps", 1, "num ps")
tf.app.flags.DEFINE_integer("steps", 1000, "steps")
tf.app.flags.DEFINE_integer("rdma", 0, "rdma")
tf.app.flags.DEFINE_integer("tensorboard", 1, "tensorboard")
tf.app.flags.DEFINE_string("tb_path", TENSORBOARD, "tensorboard log file path")
sc = SparkContext(conf=SparkConf().setAppName('data_trans')
                  .setMaster('spark://192.168.209.128:7077')
                  # .setMaster('spark://localhost:7077')
                  # .setMaster('local[*]')
                  .set("spark.jars", join(GITHUB, "TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar")))
ARGS = namedtuple("args", ['batch_size', 'mode', 'steps', 'model_path', 'rdma', 'tb_path'])
sys.path.append('/home/wjl/github/tfos')


def save_data(data):
    output_path = FLAGS.output_path
    output_example(sc, data, output_path)


def load_data():
    input_path = FLAGS.input_path
    return input_example(sc, input_path)


def op_model(rdd):
    model_path = FLAGS.model_path
    mode = FLAGS.mode
    params = ARGS._make([FLAGS.batch_size, mode, FLAGS.steps, model_path, FLAGS.rdma, FLAGS.tb_path])
    logging.error(params._asdict())
    cluster = TFCluster.run(sc, logistic_regression_dist.map_fun, params, FLAGS.cluster_size,
                            FLAGS.num_ps, FLAGS.tensorboard, TFCluster.InputMode.SPARK)
    if mode == "train":
        cluster.train(rdd, FLAGS.epochs)
    else:
        label_rdd = cluster.inference(rdd)
        print(label_rdd.take(10))
        label_rdd.saveAsTextFile(model_path)
    cluster.shutdown()


def main(unused_args):
    data = load_iris()
    save_data(data)
    rdd = load_data()
    op_model(rdd)


if __name__ == "__main__":
    tf.app.run()

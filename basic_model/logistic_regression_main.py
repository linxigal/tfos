# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 20:15
:File    : logistic_regression_main.py
"""

from os.path import join

import numpy as np
import tensorflow as tf
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from sklearn.datasets import load_iris
from tensorflowonspark import TFCluster

from basic_model import GITHUB, ROOT_PATH, logistic_regression_dist

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("output_path", join(ROOT_PATH, "output_data/iris.tfr"), "输出文件路径")
tf.app.flags.DEFINE_string("input_path", join(ROOT_PATH, "output_data/iris.tfr"), "输入文件路径")
tf.app.flags.DEFINE_string("model_path", join(ROOT_PATH, "output_data/logistic_regression_model"), "训练模型保存路径")
tf.app.flags.DEFINE_string("mode", "train", "train|inference")
tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
tf.app.flags.DEFINE_integer("cluster_size", 3, "num executor")
tf.app.flags.DEFINE_integer("epochs", 1, "num epochs")
tf.app.flags.DEFINE_integer("num_ps", 1, "num ps")
tf.app.flags.DEFINE_integer("steps", 1000, "steps")
tf.app.flags.DEFINE_integer("rdma", 0, "rdma")
tf.app.flags.DEFINE_integer("tensorboard", 1, "tensorboard")

sc = SparkContext(conf=SparkConf().setAppName('data_trans')
                  .set("spark.jars", join(GITHUB, "TensorFlowOnSpark/lib/tensorflow-hadoop-1.0-SNAPSHOT.jar")))

sqlContext = SQLContext(sc)


def to_tf_example(train_data, target_data):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "data": tf.train.Feature(float_list=tf.train.FloatList(value=train_data.astype('float64'))),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[target_data.astype('int64')])),
            }
        )
    )
    return example.SerializeToString()


def from_tf_example(byte_str):
    example = tf.train.Example()
    example.ParseFromString(byte_str)
    features = example.features.feature
    data = np.array(features['data'].float_list.value)
    label = np.array(features['label'].int64_list.value)
    return data, label


def output_example(data, output_path):
    train_data = data['data']
    train_target = data['target']

    train_rdd = sc.parallelize(train_data, 4)
    target_rdd = sc.parallelize(train_target, 4)

    tf_rdd = train_rdd.zip(target_rdd).map(lambda x: (bytearray(to_tf_example(x[0], x[1])), None))
    tf_rdd.saveAsNewAPIHadoopFile(output_path, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                  keyClass="org.apache.hadoop.io.BytesWritable",
                                  valueClass="org.apache.hadoop.io.NullWritable")


def input_example(input_path):
    tf_rdd = sc.newAPIHadoopFile(input_path, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                 keyClass="org.apache.hadoop.io.BytesWritable",
                                 valueClass="org.apache.hadoop.io.NullWritable")

    rdd = tf_rdd.map(lambda x: from_tf_example(bytes(x[0])))
    return rdd


def main(unused_args):
    data = load_iris()
    output_path = FLAGS.output_path
    input_path = FLAGS.input_path
    model_path = FLAGS.model_path
    mode = FLAGS.mode
    cluster_size = FLAGS.cluster_size
    epochs = FLAGS.epochs
    num_ps = FLAGS.num_ps
    tensorboard = FLAGS.tensorboard
    output_example(data, output_path)
    rdd = input_example(input_path)
    args = {
        'batch_size': FLAGS.batch_size,
        'mode': mode,
        'steps': FLAGS.steps,
        'model_path': model_path,
        'rdma': FLAGS.rdma
    }

    cluster = TFCluster.run(sc, logistic_regression_dist.map_fun, args, cluster_size, num_ps, tensorboard,
                            TFCluster.InputMode.SPARK)
    if mode == "train":
        cluster.train(rdd, epochs)
    else:
        label_rdd = cluster.inference(rdd)
        label_rdd.saveAsTextFile(model_path)
    cluster.shutdown()


if __name__ == "__main__":
    tf.app.run()

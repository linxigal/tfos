#!/usr/bin/env python
#-*- coding:utf-8 _*-  
"""
-------------------------------------------------
@author:weijinlong
@contact: jinlong.wei@zzjunzhi.com
@file: op_example.py
@time: 2019/5/20 13:47
-------------------------------------------------
Change Activity:
    2019/5/20:
-------------------------------------------------  
"""

import os
import shutil
import numpy as np
import tensorflow as tf


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


def output_example(sc, data, output_path):
    train_data = data['data']
    train_target = data['target']

    train_rdd = sc.parallelize(train_data, 4)
    target_rdd = sc.parallelize(train_target, 4)

    tf_rdd = train_rdd.zip(target_rdd).map(lambda x: (bytearray(to_tf_example(x[0], x[1])), None))
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    tf_rdd.saveAsNewAPIHadoopFile(output_path, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                  keyClass="org.apache.hadoop.io.BytesWritable",
                                  valueClass="org.apache.hadoop.io.NullWritable")


def input_example(sc, input_path):
    tf_rdd = sc.newAPIHadoopFile(input_path, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                 keyClass="org.apache.hadoop.io.BytesWritable",
                                 valueClass="org.apache.hadoop.io.NullWritable")

    rdd = tf_rdd.map(lambda x: from_tf_example(bytes(x[0])))
    return rdd

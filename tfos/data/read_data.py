#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 15:47
:File   : read_data.py
"""
import os
import tensorflow as tf


def tfr2sample(byte_str):
    example = tf.train.Example()
    example.ParseFromString(byte_str)
    features = example.features.feature
    image = list(features['image'].int64_list.value)
    label = list(features['label'].int64_list.value)
    return image, label


def from_csv(x):
    return [int(s) for s in x.split(',') if len(x) > 0]


class DataSet(object):

    def __init__(self, sc):
        self.sc = sc

    def read_data(self, data_dir, sub_dir=None, data_format=None):
        if sub_dir is None:
            sub_dir = ['images', 'labels']
        input_features = os.path.join(data_dir, sub_dir[0])
        input_labels = os.path.join(data_dir, sub_dir[1])
        if data_format == 'pickle':
            image_rdd = self.sc.pickleFile(input_features)
            label_rdd = self.sc.pickleFile(input_labels)
            rdd = image_rdd.zip(label_rdd)
        elif data_format == 'csv':
            image_rdd = self.sc.textFile(input_features).map(from_csv)
            label_rdd = self.sc.textFile(input_labels).map(from_csv)
            rdd = image_rdd.zip(label_rdd)
        else:
            tfr_rdd = self.sc.newAPIHadoopFile(data_dir, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                               keyClass="org.apache.hadoop.io.BytesWritable",
                                               valueClass="org.apache.hadoop.io.NullWritable")
            rdd = tfr_rdd.map(lambda x: tfr2sample(bytes(x[0])))
        return rdd.toDF(['features', 'label'])

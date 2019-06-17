#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time   : 2019/6/17 10:37
:File   : test_read_mnist.py
"""

import numpy as np
import tensorflow as tf
from examples.base import *


def fromTFExample(bytestr):
    """Deserializes a TFExample from a byte string"""
    example = tf.train.Example()
    example.ParseFromString(bytestr)
    return example


def tfr2numpy(bytestr):
    example = tf.train.Example()
    example.ParseFromString(bytestr)
    features = example.features.feature
    image = np.array(features['image'].int64_list.value)
    label = np.array(features['label'].int64_list.value)
    return image, label


class TestReadMnist(Base):
    def __init__(self, output_rdd_name, input_path, format='tfr'):
        super(TestReadMnist, self).__init__()
        self.p('output_rdd_name', output_rdd_name)
        self.p('input_path', input_path)
        self.p('format', format)

    def run(self):
        param = self.params
        output_rdd_name = param.get('output_rdd_name')

        # param = json.loads('<#zzjzParam#>')
        input_path = param.get('input_path')
        format = param.get('format')

        input_images = input_path + "/images"
        input_labels = input_path + "/labels"

        if format == 'pickle':
            image_rdd = sc.pickleFile(input_images)
            label_rdd = sc.pickleFile(input_labels)
            rdd = image_rdd.zip(label_rdd)
        elif format == 'csv':
            from_csv = lambda x: [float(s) for s in x.split(',') if len(x) > 0]
            image_rdd = sc.textFile(input_images).map(from_csv)
            label_rdd = sc.textFile(input_labels).map(from_csv)
            rdd = image_rdd.zip(label_rdd)
        else:
            # requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
            tfr_rdd = sc.newAPIHadoopFile(input_path, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                          keyClass="org.apache.hadoop.io.BytesWritable",
                                          valueClass="org.apache.hadoop.io.NullWritable")
            rdd = tfr_rdd.map(lambda x: tfr2numpy(bytes(x[0])))

        outputRDD(output_rdd_name, rdd)
        # outputRDD('<#zzjzRddName#>', rdd)


if __name__ == "__main__":
    TestReadMnist('<#zzjzRddName#>_mnist_tfr', '/home/wjl/github/tfos/output_data/mnist/tfr/train', 'tfr').run()
#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time   : 2019/6/17 10:37
:File   : test_read_mnist.py
"""

from examples.base import *


class TestReadMnist(Base):
    def __init__(self, output_rdd_name, input_path, format='tfr'):
        super(TestReadMnist, self).__init__()
        self.p('output_rdd_name', output_rdd_name)
        self.p('input_path', input_path)
        self.p('format', format)

    def run(self):
        param = self.params
        output_rdd_name = param.get('output_rdd_name')

        import tensorflow as tf

        def tfr2numpy(bytestr):
            example = tf.train.Example()
            example.ParseFromString(bytestr)
            features = example.features.feature
            image = list(features['image'].int64_list.value)
            label = list(features['label'].int64_list.value)
            return image, label

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
            from_csv = lambda x: [int(s) for s in x.split(',') if len(x) > 0]
            image_rdd = sc.textFile(input_images).map(from_csv)
            label_rdd = sc.textFile(input_labels).map(from_csv)
            rdd = image_rdd.zip(label_rdd)
        else:
            # requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
            tfr_rdd = sc.newAPIHadoopFile(input_path, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                          keyClass="org.apache.hadoop.io.BytesWritable",
                                          valueClass="org.apache.hadoop.io.NullWritable")
            rdd = tfr_rdd.map(lambda x: tfr2numpy(bytes(x[0])))

        output_df = rdd.toDF(['features', 'label'])
        outputRDD(output_rdd_name, output_df)
        # outputRDD('<#zzjzRddName#>', output_df)


if __name__ == "__main__":
    output_data_name = "<#zzjzRddName#>_mnist_tfr"
    input_path = "/home/wjl/github/tfos/output_data/mnist/tfr/train"
    TestReadMnist(output_data_name, input_path, 'tfr').run()

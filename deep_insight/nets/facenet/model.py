#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/6 10:23
:File   :facenet.py
:content:
  
"""

import unittest
import tensorflow as tf

from deep_insight import *
from deep_insight.base import *


class FaceNetSoftMax(Base):

    def __init__(self, image_size, keep_probability='1.0', embedding_size='128', weight_decay='0.0'):
        super(FaceNetSoftMax, self).__init__()
        self.p('image_size', image_size)
        # self.p('n_classes', n_classes)
        self.p('keep_probability', keep_probability)
        self.p('embedding_size', embedding_size)
        self.p('weight_decay', weight_decay)

    def run(self):
        param = self.params

        from tfos.nets.facenet.model import FaceNetSoftMax

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        image_size = param.get('image_size')
        n_classes = param.get('n_classes')
        keep_probability = param.get('keep_probability', 3)
        embedding_size = param.get('embedding_size', 3)
        weight_decay = param.get('weight_decay', 3)

        # load data
        assert input_rdd_name is not None, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"

        # kwargs = dict(image_size=int(image_size), n_classes=int(n_classes))
        kwargs = dict(image_size=int(image_size), n_classes=input_rdd.count())
        if keep_probability:
            keep_probability = float(keep_probability)
            assert 0 < keep_probability <= 1, "0 < keep_probability <= 1"
            kwargs['keep_probability'] = keep_probability
        if embedding_size:
            kwargs['embedding_size'] = int(embedding_size)
        if weight_decay:
            weight_decay = float(weight_decay)
            assert 0 < keep_probability <= 1, "0 < weight_decay <= 1"
            kwargs['weight_decay'] = weight_decay
        output_df = FaceNetSoftMax(**kwargs).build_model().serialize(sqlc)
        output_df.withColumn('data', input_rdd_name)
        output_df.show()
        outputRDD('<#zzjzRddName#>_facenet', output_df)


class TestFaceNet(unittest.TestCase):
    def setUp(self) -> None:
        self.is_local = True
        self.input_dir = os.path.join(self.path, "data/data/lfw")
        self.output_dir = os.path.join(self.path, "data/data/lfw_160_30")

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    def test_facenet_softmax(self):
        FaceNetSoftMax(image_size='160').run()
        graph = tf.get_default_graph().as_graph_def()
        # print(tf.train.export_meta_graph(graph=graph))


if __name__ == '__main__':
    unittest.main()

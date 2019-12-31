#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/21 17:00
:File   :train.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *


class FaceNetSoftMaxTrain(Base):
    def __init__(self, cluster_size, num_ps, batch_size, epochs, model_dir, go_on='false'):
        super(FaceNetSoftMaxTrain, self).__init__()
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        self.p('model_dir', model_dir)
        self.p('go_on', go_on)

    def run(self):
        param = self.params

        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool
        from tfos.nets.facenet.tfos import TFOSFaceNetSoftMax

        # param = json.loads('<#zzjzParam#>')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 3)
        batch_size = param.get('batch_size', 3)
        epochs = param.get('epochs', 3)
        model_dir = param.get('model_dir', 3)
        go_on = param.get('go_on', BOOLEAN[0])

        cluster_size = int(cluster_size)
        num_ps = int(num_ps)

        kwargs = dict(batch_size=int(batch_size), epochs=int(epochs), model_dir=model_dir)
        if go_on:
            kwargs['go_on'] = convert_bool(go_on)

        output_df = TFOSFaceNetSoftMax(sc, sqlc, cluster_size, num_ps).train(**kwargs)
        outputRDD('<#zzjzRddName#>_Masking', output_df)


class TestFaceNetSoftMaxTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.is_local = True
        self.input_dir = os.path.join(self.path, "data/data/lfw")
        self.output_dir = os.path.join(self.path, "data/data/lfw_160_30")

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    def test_facenet_softmax_train(self):
        FaceNetSoftMaxTrain().run()


if __name__ == '__main__':
    unittest.main()

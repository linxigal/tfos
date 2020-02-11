#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/12/31 11:06
:File   :mtcnn.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *


class MTCNN(Base):
    def __init__(self, cluster_size, input_dir, output_dir, image_size='182', margin='44', random_order='true',
                 gpu_memory_fraction='1.0', detect_multiple_faces='false'):
        super(MTCNN, self).__init__()
        self.p('cluster_size', cluster_size)
        self.p('input_dir', input_dir)
        self.p('output_dir', output_dir)
        self.p('image_size', image_size)
        self.p('margin', margin)
        self.p('random_order', random_order)
        self.p('gpu_memory_fraction', gpu_memory_fraction)
        self.p('detect_multiple_faces', detect_multiple_faces)

    def run(self):
        param = self.params
        from tfos.choices import BOOLEAN
        from tfos.utils import convert_bool
        from tfos.nets.facenet import TFOS_MTCNN

        # param = json.loads('<#zzjzParam#>')
        cluster_size = param.get('cluster_size', 3)
        input_dir = param.get('input_dir')
        output_dir = param.get('output_dir')
        image_size = param.get('image_size', 160)
        margin = param.get('margin', 30)
        random_order = param.get('random_order', BOOLEAN[0])
        gpu_memory_fraction = param.get('gpu_memory_fraction', 1.0)
        detect_multiple_faces = param.get('detect_multiple_faces', BOOLEAN[1])

        cluster_size = int(cluster_size)
        kwargs = dict(input_dir=input_dir, output_dir=output_dir)
        if image_size:
            kwargs['image_size'] = int(image_size)
        if margin:
            kwargs['margin'] = int(margin)
        if random_order:
            kwargs['random_order'] = convert_bool(random_order)
        if gpu_memory_fraction:
            kwargs['gpu_memory_fraction'] = float(gpu_memory_fraction)
        if detect_multiple_faces:
            kwargs['detect_multiple_faces'] = convert_bool(detect_multiple_faces)

        output_df = TFOS_MTCNN(sc, cluster_size=cluster_size).run(**kwargs)
        outputRDD('<#zzjzRddName#>_data', output_df)


class TestMTCNN(unittest.TestCase):
    def setUp(self) -> None:
        self.is_local = True
        self.input_dir = os.path.join(self.path, "data/data/lfw/lfw")
        self.output_dir = os.path.join(self.path, "data/data/lfw/lfw_160_30")

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    def test_mtcnn(self):
        MTCNN(cluster_size=3,
              input_dir=self.input_dir,
              output_dir=self.output_dir).run()


if __name__ == '__main__':
    unittest.main()

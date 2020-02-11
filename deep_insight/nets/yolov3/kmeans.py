#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/27 11:03
:File   :kmeans.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *


class YOLOV3KMeans(Base):
    """
    参数：
        cluster_number: 聚类K值
            聚类的类别数量
        label_path: 训练数据集
            待训练数据集的路径集合文件
    """

    def __init__(self, cluster_number, label_path):
        super(YOLOV3KMeans, self).__init__()
        self.p('cluster_number', cluster_number)
        self.p('label_path', label_path)

    def run(self):
        param = self.params

        from tfos.nets.yolov3 import YOLOV3KMeansLayer

        # param = json.loads('<#zzjzParam#>')
        cluster_number = param.get('cluster_number', '9')
        label_path = param.get('label_path')

        YOLOV3KMeansLayer(cluster_number, label_path).txt2clusters()


class TestYOLOV3KMeans(unittest.TestCase):
    def setUp(self):
        self.is_local = False
        self.cluster_number = 9

    @property
    def path(self):
        if self.is_local:
            local_path = '/home/wjl/github/keras-yolo3/VOCdevkit/VOC2007/train/train.txt'
            return local_path
        else:
            data_dir = 'data/data/yolov3/VOCdevkit/VOC2007/train/train.txt'
            return os.path.join(HDFS, data_dir)

    # @unittest.skip('')
    def test_yolov3_kmeans(self):
        YOLOV3KMeans(self.cluster_number, self.path).run()


if __name__ == '__main__':
    unittest.main()

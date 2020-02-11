#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/26 9:32
:File   :voc_label.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *


class VOCLabel(Base):
    """
     参数：
        data_dir: 数据集目录
            训练数据集的目录
        output_dir: 输出目录
            根据数据处理后生成的文件的存放目录
        image_format: 图像格式
            数据集中图像的格式
    """

    def __init__(self, data_dir, output_dir, image_format='jpg'):
        super(VOCLabel, self).__init__()
        self.p('data_dir', data_dir)
        self.p('output_dir', output_dir)
        self.p('image_format', image_format)

    def run(self):
        param = self.params

        from tfos.nets.yolov3 import VOCLabelLayer
        from tfos.choices import IMG_FORMAT

        # param = json.loads('<#zzjzParam#>')
        data_dir = param.get('data_dir')
        output_dir = param.get('output_dir')
        image_format = param.get('image_format', IMG_FORMAT[0])

        VOCLabelLayer(data_dir, output_dir, image_format).do()


class TestVOCLabel(unittest.TestCase):

    def setUp(self):
        self.is_local = False
        self.image_format = 'jpg'

    @property
    def data_path(self):
        if self.is_local:
            local_path = '/home/wjl/github/keras-yolo3/VOCdevkit/VOC2007'
            return local_path
        else:
            data_dir = 'data/data/yolov3/VOCdevkit/VOC2007'
            return os.path.join(HDFS, data_dir)

    @property
    def output_path(self):
        if self.is_local:
            local_path = '/home/wjl/github/keras-yolo3/VOCdevkit/VOC2007/train'
            return local_path
        else:
            data_dir = 'data/data/yolov3/VOCdevkit/VOC2007/train'
            return os.path.join(HDFS, data_dir)

    # @unittest.skip('')
    def test_voc_label(self):
        VOCLabel(self.data_path, self.output_path, self.image_format).run()


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/25 15:14
:File   :yolo.py
:content:
  
"""

import unittest

from deep_insight.base import *


class YOLOV3Model(Base):

    def __init__(self, num_anchors, num_classes):
        super(YOLOV3Model, self).__init__()
        self.p('num_anchors', num_anchors)
        self.p('num_classes', num_classes)

    def run(self):
        param = self.params

        from tfos.nets.yolov3 import YOLOV3ModelLayer

        # param = json.loads('<#zzjzParam#>')
        num_anchors = param.get('num_anchors')
        num_classes = param.get('num_classes')

        num_anchors = int(num_anchors)
        num_classes = int(num_classes)

        output_df = YOLOV3ModelLayer(sqlc=sqlc).add(num_anchors, num_classes)
        outputRDD('<#zzjzRddName#>_yolov3_model', output_df)


class YOLOV3TinyModel(Base):

    def __init__(self, num_anchors, num_classes):
        super(YOLOV3TinyModel, self).__init__()
        self.p('num_anchors', num_anchors)
        self.p('num_classes', num_classes)

    def run(self):
        param = self.params

        from tfos.nets.yolov3 import YOLOV3TinyModelLayer

        # param = json.loads('<#zzjzParam#>')
        num_anchors = param.get('num_anchors')
        num_classes = param.get('num_classes')

        num_anchors = int(num_anchors)
        num_classes = int(num_classes)

        output_df = YOLOV3TinyModelLayer(sqlc=sqlc).add(num_anchors, num_classes)
        outputRDD('<#zzjzRddName#>_yolov3_tiny_model', output_df)


class TestYOLOV3(unittest.TestCase):

    @unittest.skip('')
    def test_yolov3_model(self):
        YOLOV3Model('9', '20').run()
        SummaryLayer().run()

    # @unittest.skip('')
    def test_yolov3_tiny_model(self):
        YOLOV3TinyModel('9', '20').run()
        SummaryLayer().run()


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/25 15:17
:File   :yolov3.py
:content:
  
"""

from tensorflow.python.keras.layers import Input

from tfos.base import BaseLayer, ext_exception
from .model import yolo_body, tiny_yolo_body


class YOLOV3ModelLayer(BaseLayer):

    @ext_exception("yolov3 model")
    def add(self, num_anchors, num_classes):
        image_input = Input(shape=(None, None, 3))
        model_body = yolo_body(image_input, num_anchors // 3, num_classes)
        return self.model2df(model_body)


class YOLOV3TinyModelLayer(BaseLayer):

    @ext_exception("yolov3 tiny model")
    def add(self, num_anchors, num_classes):
        image_input = Input(shape=(None, None, 3))
        model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
        return self.model2df(model_body)

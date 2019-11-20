#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/12/5 9:34
:File   :test.py
:content:
  
"""

import tensorflow as tf
import json
import pandas as pd
from os.path import join
from tensorflow.python.keras.layers import Input

from tfos.nets.yolov3.train import YOLOV3ModelTrainWorker, YOLOV3Method
from tfos.base import CustomEncoder
from tfos.base.gfile import ModelDir
from tfos.nets.yolov3.model import yolo_body


class TestYOLOV3ModelTrain(object):
    def __init__(self):
        self.go_on = True
        self.classes_path = join(self.path, 'data/data/yolov3/label/voc_classes.txt')
        self.anchors_path = join(self.path, 'data/data/yolov3/label/yolo_anchors.txt')
        self.weights_path = join(self.path, 'data/data/yolov3/label/yolo_weights.h5')
        self.train_path = join(self.path, 'data/data/yolov3/VOCdevkit/VOC2007/train/test_train.txt')
        self.val_path = join(self.path, 'data/data/yolov3/VOCdevkit/VOC2007/train/test_val.txt')
        self.model_dir = join(self.path, 'data/model/yolov3')
        self.model_rdd = None
        self.train_num = 1

    @property
    def path(self):
        root = 'D:\\github\\tfos'
        # root = '/home/wjl/github/tfos'
        return root

    def build_model(self):
        image_input = Input(shape=(None, None, 3))
        anchors = YOLOV3Method.get_anchors(self.anchors_path)
        classes = YOLOV3Method.get_classes(self.classes_path)
        num_anchors = len(anchors)
        num_classes = len(classes)
        model_body = yolo_body(image_input, num_anchors // 3, num_classes)
        self.model_rdd = pd.DataFrame([{'model_config': json.dumps(model_body.get_config(), cls=CustomEncoder)}])

    def read_train_set(self):
        with tf.io.gfile.GFile(self.train_path, 'r') as val_file:
            train_data = val_file.readlines()
            self.train_num = len(train_data)
            return train_data

    def train_generate_data(self, train_data):
        pass

    def test_yolov3_model_train(self):
        md = ModelDir(self.model_dir, 'train*')
        if self.go_on:
            md.create_model_dir()
        else:
            md = md.rebuild_model_dir()
        self.build_model()
        YOLOV3ModelTrainWorker(
            model_rdd=self.model_rdd,
            batch_size=32,
            epochs=1,
            classes_path=self.classes_path,
            anchors_path=self.anchors_path,
            # weights_path=self.weights_path,
            # train_path=self.train_path,
            val_path=self.val_path,
            image_size=(416, 416),
            go_on=self.go_on,
            **md.to_dict()
        )()


if __name__ == '__main__':
    TestYOLOV3ModelTrain().test_yolov3_model_train()

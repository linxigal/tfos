#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/12/5 9:34
:File   :test.py
:content:
  
"""

import os
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam

from tfos import ROOT_PATH
from tfos.base import logger
from tfos.base.gfile import ModelDir
from tfos.nets.yolov3.model import yolo_body, yolo_loss, preprocess_true_boxes
from tfos.nets.yolov3.utils import get_random_data


class YOLOV3Method(object):

    @staticmethod
    def get_classes(classes_path):
        '''loads the classes'''
        with tf.io.gfile.GFile(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def get_anchors(anchors_path):
        '''loads the anchors from a file'''
        with tf.io.gfile.GFile(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        '''data generator for fit_generator'''
        n = len(annotation_lines)
        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                dest_path = annotation_lines[i].split()[0]
                # if is_copy:
                #     dest_path = self.copy_image(dest_path)
                image, box = get_random_data(dest_path, input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data] + y_true, np.zeros(batch_size)

    def data_generator_wrapper(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0:
            return None
        return self.data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

    # def copy_image(self, source_path):
    #     filename = source_path.split('/')[-1]
    #     dest_path = os.path.join(self.tmp_dir, filename)
    #     tf.io.gfile.copy(source_path, dest_path, True)
    #     return dest_path


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
        self.batch_size = 4
        self.anchors = YOLOV3Method.get_anchors(self.anchors_path)
        self.classes = YOLOV3Method.get_classes(self.classes_path)
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.classes)
        self.input_shape = (416, 416)
        self.model = None
        self.name = 'yolov3'
        self.initial_epoch = 0

    @property
    def path(self):
        # root = 'D:\\github\\tfos'
        # root = '/home/wjl/github/tfos'
        root = ROOT_PATH
        return root

    def build_model(self):
        image_input = Input(shape=(None, None, 3))
        h, w = self.input_shape
        num_anchors = len(self.anchors)
        model = yolo_body(image_input, self.num_anchors // 3, self.num_classes)
        y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                               num_anchors // 3, self.num_classes + 5), name='tmp_{}'.format(l)) for l in range(3)]
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': self.anchors,
                                       'num_classes': self.num_classes,
                                       'ignore_thresh': 0.5},
                            trainable=False)(model.output + y_true)
        model = Model([model.input] + y_true, model_loss, trainable=False)
        self.model = model

    @property
    def train_set(self):
        with tf.io.gfile.GFile(self.train_path, 'r') as f:
            train_data = f.readlines()
            return train_data

    def train_generate_data(self, lines):
        ym = YOLOV3Method()
        return ym.data_generator_wrapper(lines, self.batch_size, self.input_shape, self.anchors, self.num_classes)

    @staticmethod
    def get_results(his):
        results = []
        length = 0
        for key, values in his.history.items():
            length = len(values)
            results.append(zip([key] * len(values), values))
        results.append(zip(['_epoch'] * length, his.epoch))
        return [dict(v) for v in zip(*results)]

    def restore_model(self, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            K.set_learning_phase(False)
            self.initial_epoch = int(ckpt.model_checkpoint_path.split('_')[-1])
            self.model.load_weights(ckpt.model_checkpoint_path)

    def train(self, save_dir, result_dir, checkpoint_dir, log_dir):
        result_file = os.path.join(result_dir, "train_result.txt")
        train_set = self.train_set
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as sess:
            K.set_session(sess)
            if self.go_on:
                self.restore_model(checkpoint_dir)
            tb_callback = TensorBoard(log_dir=log_dir, write_images=True)
            checkpoint_file = os.path.join(checkpoint_dir, self.name + '_checkpoint_{epoch}')
            ckpt_callback = ModelCheckpoint(checkpoint_file,
                                            monitor='loss',
                                            save_weights_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
            early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            callbacks = [tb_callback, ckpt_callback]
            # callbacks = []

            self.model.compile(optimizer=Adam(lr=1e-4),
                               loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            # print('Unfreeze all of the layers.')
            callbacks.extend([reduce_lr, early_stopping])
            steps_per_epoch = len(train_set) // self.batch_size
            # note that more GPU memory is required after unfreezing the body
            # try:
            his = self.model.fit_generator(self.train_generate_data(train_set),
                                           steps_per_epoch=steps_per_epoch,
                                           # validation_data=self.val_generate_data(val_data),
                                           # validation_steps=max(1, self.val_num // self.batch_size),
                                           epochs=self.initial_epoch + 1,
                                           initial_epoch=self.initial_epoch,
                                           workers=1,
                                           callbacks=callbacks)
            logger.debug(str(his.history))
            # except Exception as e:
            #     logger.debug(str(e))
            logger.debug('end')
            save_model_path = os.path.join(save_dir, 'model.h5')
            self.model.save(save_model_path)
            ModelDir.write_result(result_file, self.get_results(his))

    def main(self):
        md = ModelDir(self.model_dir, 'train*')
        if self.go_on:
            md.create_model_dir()
        else:
            md = md.rebuild_model_dir()
        self.build_model()
        self.train(**md.to_dict())


if __name__ == '__main__':
    TestYOLOV3ModelTrain().main()

#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/25 16:40
:File   :train.py
:content:
  
"""

import colorsys
import json
import os

import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.layers import Lambda, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, RMSprop, SGD

from tfos.base.gfile import ModelDir
from tfos.worker import Worker, logger
from .model import preprocess_true_boxes, yolo_loss, yolo_eval
from .utils import get_random_data, letterbox_image


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

    def data_generator(self, annotation_lines, batch_size, input_shape, anchors, num_classes, is_copy=True):
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
                if is_copy:
                    dest_path = self.copy_image(dest_path)
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

    def copy_image(self, source_path):
        filename = source_path.split('/')[-1]
        dest_path = os.path.join(self.tmp_dir, filename)
        tf.io.gfile.copy(source_path, dest_path, True)
        return dest_path


class YOLOV3Worker(Worker, YOLOV3Method):
    def __init__(self, model_rdd, classes_path, anchors_path, go_on, val_path, image_size,
                 weights_path=None, freeze_body=2, *args, **kwargs):
        super(YOLOV3Worker, self).__init__(*args, **kwargs)
        self.model_config = json.loads(model_rdd.first().model_config)
        self.class_names = self.get_classes(classes_path)
        self.num_classes = len(self.class_names)
        # self.anchors = self.get_anchors(anchors_path)[:6]
        self.anchors = self.get_anchors(anchors_path)
        self.weights_path = weights_path
        self.go_on = go_on
        self.val_path = val_path
        self.image_size = image_size
        self.freeze_body = freeze_body
        self.initial_epoch = 0
        self.val_num = 1

    def generate_rdd_data(self):
        while not self.tf_feed.should_stop():
            batches = self.tf_feed.next_batch(self.batch_size)
            image_data = []
            box_data = []
            # if not batches or len(batches) < self.batch_size:
            if not batches:
                raise StopIteration()
            for row in batches:
                image, box = get_random_data(self.copy_image(row.split()[0]), self.image_size, random=True)
                image_data.append(image)
                box_data.append(box)
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, self.image_size, self.anchors, self.num_classes)
            yield [image_data] + y_true, None

    def load_weights(self):
        if self.weights_path:
            weight_filename = self.weights_path.split('/')[-1]
            tmp_filename = os.path.join(self.tmp_dir, weight_filename)
            tf.io.gfile.copy(self.weights_path, tmp_filename, True)
            self.model.load_weights(tmp_filename)

    def save_weights(self, name):
        if self.task_index == 0:
            tmp_path = os.path.join(self.tmp_dir, name)
            save_path = os.path.join(self.save_dir, name)
            self.model.save_weights(tmp_path)
            tf.io.gfile.copy(tmp_path, save_path, True)

    def read_val_set(self):
        with tf.io.gfile.GFile(self.val_path, 'r') as val_file:
            val_data = val_file.readlines()
            self.val_num = len(val_data)
            return val_data

    def val_generate_data(self, val_data):
        return self.data_generator_wrapper(val_data, self.batch_size, self.image_size,
                                           self.anchors, self.num_classes)

    def get_results(self, his):
        results = []
        length = 0
        for key, values in his.history.items():
            length = len(values)
            results.append(zip([key] * len(values), values))
        results.append(zip(['_task_index'] * length, [self.task_index] * length))
        results.append(zip(['_epoch'] * length, his.epoch))
        return [dict(v) for v in zip(*results)]

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # K.set_learning_phase(False)
            self.initial_epoch = int(ckpt.model_checkpoint_path.split('_')[-1])
            self.model.load_weights(ckpt.model_checkpoint_path)

    def execute(self):
        result_file = os.path.join(self.result_dir, "train_result_{}.txt".format(self.task_index))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(self.server.target, config=config) as sess:
            K.set_session(sess)
            # K.set_learning_phase(False)
            if self.go_on:
                self.restore_model()
            tb_callback = TensorBoard(log_dir=self.log_dir, write_grads=True, write_images=True)
            ckpt_callback = ModelCheckpoint(self.checkpoint_file,
                                            monitor='loss',
                                            save_weights_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
            early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            callbacks = [tb_callback, ckpt_callback] if self.task_index == 0 else []

            self.model.compile(optimizer=Adam(lr=1e-4),
                               loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            callbacks.extend([reduce_lr, early_stopping])
            try:
                logger.debug('start')
                # val_data = self.read_val_set()
                his = self.model.fit_generator(self.generate_rdd_data(),
                                               steps_per_epoch=self.steps_per_epoch,
                                               # validation_data=self.val_generate_data(val_data),
                                               # validation_steps=max(1, self.val_num // self.batch_size),
                                               epochs=self.epochs + self.initial_epoch,
                                               initial_epoch=self.initial_epoch,
                                               workers=1,
                                               callbacks=callbacks)
                logger.debug(his.history)
            except Exception as e:
                logger.debug(str(e))
            logger.debug('end')
            self.save_model()
            ModelDir.write_result(result_file, self.get_results(his), self.go_on)
            self.tf_feed.terminate()
        K.clear_session()


class YOLOV3ModelTrainWorker(YOLOV3Worker):

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            h, w = self.image_size
            num_anchors = len(self.anchors)
            y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                                   num_anchors // 3, self.num_classes + 5), name='tmp_{}'.format(l)) for l in range(3)]
            model = Model.from_config(self.model_config)
            model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                arguments={'anchors': self.anchors,
                                           'num_classes': self.num_classes,
                                           'ignore_thresh': 0.5})(model.output + y_true)
            self.model = Model([model.input] + y_true, model_loss)


class YOLOV3TinyModelTrainWorker(YOLOV3Worker):
    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            h, w = self.image_size
            num_anchors = len(self.anchors)
            y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                                   num_anchors // 2, self.num_classes + 5), name='tmp_{}'.format(l)) for l in range(2)]
            model = Model.from_config(self.model_config)
            # model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
            model_loss = Lambda(yolo_loss, output_shape=(), name='yolo_loss',
                                arguments={'anchors': self.anchors,
                                           'num_classes': self.num_classes,
                                           'ignore_thresh': 0.3})(model.output + y_true)
                                           # 'ignore_thresh': 0.7})(model.output + y_true)
            model = Model([model.input] + y_true, model_loss)
            self.model = model


class YOLOV3PredictWorker(YOLOV3Worker):

    def __init__(self, score, iou, **kwargs):
        super(YOLOV3Worker, self).__init__(**kwargs)
        self.score = score
        self.iou = iou
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def build_model(self):
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            self.load_model()

    def execute(self):
        with tf.Session(self.server.target) as sess:
            K.set_session(sess)
            self.load_model()
            image = None
            if self.image_size != (None, None):
                assert self.image_size[0] % 32 == 0, 'Multiples of 32 required'
                assert self.image_size[1] % 32 == 0, 'Multiples of 32 required'
                boxed_image = letterbox_image(image, tuple(reversed(self.image_size)))
            else:
                new_image_size = (image.width - (image.width % 32),
                                  image.height - (image.height % 32))
                boxed_image = letterbox_image(image, new_image_size)
            image_data = np.array(boxed_image, dtype='float32')

            print(image_data.shape)
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)
            boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                               len(self.class_names), self.image_size,
                                               score_threshold=self.score, iou_threshold=self.iou)
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.model.input: image_data,
                    self.image_size: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })
            return boxes, scores, classes

    def draw_box(self, image, out_boxes, out_scores, out_classes):
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image

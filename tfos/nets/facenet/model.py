#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/2 10:28
:File   :facenet.py
:content:
  
"""
import tensorflow as tf
from facenet.src import facenet
from facenet.src.models.inception_resnet_v1 import inference
from tensorflow.python.ops import data_flow_ops

from tfos.tf import add_collection, TFMode
from tfos.tf.model import INPUTS, OUTPUTS


class FaceNetSoftMax(TFMode):
    def __init__(self, image_size, n_classes, keep_probability=1.0, embedding_size=128, weight_decay=0.0):
        super(FaceNetSoftMax, self).__init__()
        self.image_size = (image_size, image_size)
        self.keep_probability = keep_probability
        self.embedding_size = embedding_size
        self.weight_decay = weight_decay
        self.add_params(weight_decay=weight_decay, image_size=image_size, n_classes=n_classes)

    def build_model(self):
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
        self.add_inputs(batch_size_placeholder, phase_train_placeholder)
        input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                              dtypes=[tf.string, tf.int32, tf.int32],
                                              shapes=[(1,), (1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many(
            [image_paths_placeholder, labels_placeholder, control_placeholder],
            name='enqueue_op')

        add_collection(INPUTS, image_paths_placeholder, labels_placeholder, control_placeholder)
        add_collection(OUTPUTS, enqueue_op)

        images_and_labels = []
        nrof_preprocess_threads = 4
        images_and_labels = facenet.create_input_pipeline(images_and_labels, input_queue, self.image_size,
                                                          nrof_preprocess_threads)

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[self.image_size + (3,), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * 100,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        # Build the inference graph
        prelogits, _ = inference(image_batch, self.keep_probability,
                                 phase_train=phase_train_placeholder,
                                 bottleneck_layer_size=self.embedding_size,
                                 weight_decay=self.weight_decay)
        self.add_outputs(prelogits, label_batch)
        return self

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 10:21
:File       : tf_read.py
"""

import tensorflow as tf
import numpy as np


def valid_filename_queue(filename):
    if isinstance(filename, str):
        filename = [filename]
    for f in filename:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(filename)
    return filename_queue


def from_tensor_slices(raw, batch_size, num_workers=None):
    data = tf.data.Dataset.from_tensor_slices(raw)
    if num_workers:
        data = data.shard(num_workers)
    data = data.batch(batch_size)
    return data.make_one_shot_iterator()


def from_generator(generator, batch_size, num_workers=None):
    data = tf.data.Dataset.from_generator(generator)
    if num_workers:
        data = data.shard(num_workers)
    data = data.batch(batch_size)
    return data.make_one_shot_iterator()


def read_csv(filename, batch_size=1):
    filename_queue = valid_filename_queue(filename)
    reader = tf.TextLineReader(filename_queue)
    record = reader.read()
    raw = tf.decode_csv(record, tf.float64)
    return from_tensor_slices(raw, batch_size)


def read_bin(filename, batch_size=1):
    filename_queue = valid_filename_queue(filename)
    reader = tf.FixedLengthRecordReader(filename_queue)
    record = reader.read(filename_queue)
    raw = tf.decode_raw(record, tf.float64)
    return from_tensor_slices(raw, batch_size)


def read_tfr(filename_queue, batch_size=1):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'x': tf.FixedLenFeature([], tf.string),
                                           'y': tf.FixedLenFeature([], tf.int64),
                                       })
    x = tf.decode_raw(features['x'], tf.float64)
    y = tf.cast(features['y'], tf.int32)
    return x, y


def read_img(filename, dimensions, batch_size=1):
    filename_queue = valid_filename_queue(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, dimensions)
    label = tf.cast(features['label'], tf.int32)
    return img, label


def read_rdd(tf_feed, batch_size):
    def generator():
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(1)
            if len(batch) == 0:
                return
            row = batch[0]
            x = np.array(row[0]).astype(np.float32) / 255.0
            y = np.array(row[1]).astype(np.int64)
            yield (x, y)
    return from_generator(generator, batch_size)

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/28 16:29
:File       : read_image_dir.py
"""

import os
import tensorflow as tf


def _read_csv(filename, dims, label_column):
    """

    :param filename:
    :param dims: int, features dimensions
    :param label_column:
    :return:
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    data = tf.decode_csv(value, record_defaults=[[1.0]] * dims)
    label = int(data.pop(label_column))
    features = tf.stack(data)
    return features, label


def read_csv(filename, dims, label_column, batch_size=1):
    """

    :param filename:
    :param dims:
    :param label_column:
    :param batch_size:
    :return:
    """
    features, label = _read_csv(filename, dims, label_column)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    batch_image, batch_label = tf.train.shuffle_batch([features, label], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return batch_image, batch_label


def read_image_dir(image_dir: str, dims: list):
    """

    :param image_dir:
    :param dims:
    :return: tuple (x, y)
    """


def _read_tfr(filename_queue, dims):
    """ read file by tfrecord format

    :param filename_queue:  file queue
    :param dims: image dimensions of list
    :return: image and label of tuple
    """
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    features = tf.parse_single_example(value, features={'label': tf.FixedLenFeature([], tf.int64),
                                                        'img': tf.FixedLenFeature([], tf.string), })
    img = tf.decode_raw(features["img"], tf.uint8)
    img = tf.reshape(img, dims)
    label = tf.cast(features["label"], tf.int32)
    return img, label


def read_tfr(filename, dims, batch_size=1):
    """ batch read tfrecord by filename queue

    :param filename: file name of tfrecord format
    :param batch_size: batch size
    :return: return a tuple contains of images and labels, the size of image and label size is batch_size
    """
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)
    img, label = _read_tfr(filename_queue, dims)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    batch_image, batch_label = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return batch_image, batch_label

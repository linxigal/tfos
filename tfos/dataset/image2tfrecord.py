#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/28 16:31
:File       : image2tfrecord.py
"""

import os
from PIL import Image
import tensorflow as tf


def _int64_feature(value):
    """
    generate int64 feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    generate byte feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image2tfrecord(image_dir, dims, tfr_dir=None, classes=[]):
    if not tfr_dir:
        tfr_dir = image_dir + '_tfr'
    if not os.path.exists(tfr_dir):
        os.makedirs(tfr_dir)
    for class_index, class_name in enumerate(classes):
        class_tfr = os.path.join(tfr_dir, class_name + '.tfrecord')
        writer = tf.python_io.TFRecordWriter(class_tfr)
        class_dir = os.path.join(image_dir, class_name)
        filenames = os.listdir(class_dir)
        for index, fn in enumerate(filenames, 1):
            image = Image.open(os.path.join(image_dir, class_name, fn))
            image = image.resize(dims)
            image_raw = image.tobytes()
            print(index, fn)

            # write image data(pixel values and label) to Example Protocol Buffer
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": _int64_feature(class_index),
                "image_raw": _bytes_feature(image_raw),
            }))

            # write an example to TFRecord file
            writer.write(example.SerializeToString())
        writer.close()


if __name__ == "__main__":
    image2tfrecord("E:\\data\\flower_photos", [240, 240], classes=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
    # image2tfrecord("E:\\data\\flower_photos", [100, 100], classes=['test'])

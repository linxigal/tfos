#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/27 14:32
:File       : image_dir.py
"""

import os
import numpy as np
import tensorflow as tf
from operator import mul
from functools import reduce


class ReadImage(object):
    def __init__(self, ctx, args, channels):
        self.ctx = ctx
        self.args = args
        self.channels = channels

    @property
    def image_size(self):
        # return int(np.prod(self.channels))
        return reduce(mul, self.channels)

    def _parse_csv(self, ln):
        splits = tf.string_split([ln], delimiter='|')
        lbl = splits.values[0]
        img = splits.values[1]
        image_defaults = [[0.0] for col in range(self.image_size)]
        image = tf.stack(tf.decode_csv(img, record_defaults=image_defaults))
        norm = tf.constant(255, dtype=tf.float32, shape=(784,))
        normalized_image = tf.div(image, norm)
        label_value = tf.string_to_number(lbl, tf.int32)
        label = tf.one_hot(label_value, 10)
        return normalized_image, label

    def _parse_tfr(self, example_proto, width, height):
        feature_def = {"label": tf.FixedLenFeature(10, tf.int64),
                       "image": tf.FixedLenFeature(self.image_size, tf.int64)}
        features = tf.parse_single_example(example_proto, feature_def)
        norm = tf.constant(255, dtype=tf.float32, shape=(784,))
        image = tf.div(tf.to_float(features['image']), norm)
        label = tf.to_float(features['label'])
        return image, label

    def image_iterator(self):
        # Dataset for input data
        image_dir = self.ctx.absolute_path(self.args.images_labels)
        file_pattern = os.path.join(image_dir, 'part-*')

        ds = tf.data.Dataset.list_files(file_pattern)
        # ds = ds.shard(num_workers, task_index).repeat(args.epochs).shuffle(args.shuffle_size)
        if self.args.format == 'csv2':
            ds = ds.interleave(tf.data.TextLineDataset, cycle_length=self.args.readers, block_length=1)
            parse_fn = self._parse_csv
        else:  # args.format == 'tfr'
            ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=self.args.readers, block_length=1)
            parse_fn = self._parse_tfr
        ds = ds.map(parse_fn).batch(self.args.batch_size)
        iterator = ds.make_one_shot_iterator()
        return iterator

#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/25 8:45
:File   :cifar.py
:content:
  
"""
import os
import sys

import numpy as np
import tensorflow as tf
from six.moves import cPickle
from tensorflow.python.keras import backend as K

from tfos.data import BaseData


def load_batch(fpath, label_key='labels'):
    with tf.gfile.Open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


class Cifar10(BaseData):
    def __init__(self, **kwargs):
        super(Cifar10, self).__init__(**kwargs)

        self.num_train_samples = 50000
        self.num_batch_samples = 10000
        self.num_test_samples = 10000
        self.batch_size = 5
        self.height = 32
        self.width = 32
        self.channel = 3
        self.num_class = 10
        self.transpose = (0, 2, 3, 1)

    @property
    def train_df(self):
        rdd = self.sc.parallelize(zip(*self.load_train()))
        return self.rdd2df(rdd)

    @property
    def test_df(self):
        rdd = self.sc.parallelize(zip(*self.load_test()))
        return self.rdd2df(rdd)

    def load_train(self):
        x_train = np.empty((self.num_train_samples, self.channel, self.height, self.width), dtype='uint8')
        y_train = np.empty((self.num_train_samples,), dtype='uint8')

        for i in range(self.batch_size):
            fpath = os.path.join(self.path, 'data_batch_' + str(i + 1))
            (x_train[i * self.num_batch_samples:(i + 1) * self.num_batch_samples, :, :, :],
             y_train[i * self.num_batch_samples:(i + 1) * self.num_batch_samples]) = load_batch(fpath)

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(*self.transpose)
        if self.one_hot:
            y_train = self.convert_one_hot(y_train, self.num_class)

        if self.flat:
            x_train = x_train.reshape((self.num_train_samples, self.height * self.width * self.channel))

        return x_train.tolist(), y_train.tolist()

    def load_test(self):
        fpath = os.path.join(self.path, 'test_batch')
        x_test, y_test = load_batch(fpath)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        if K.image_data_format() == 'channels_last':
            x_test = x_test.transpose(*self.transpose)

        if self.one_hot:
            y_test = self.convert_one_hot(y_test, self.num_class)

        if self.flat:
            x_test = x_test.reshape((self.num_test_samples, self.height * self.width * self.channel))

        return x_test.tolist(), y_test.tolist()


class Cifar100(BaseData):
    def __init__(self, label_mode='fine', **kwargs):
        super(Cifar100, self).__init__(**kwargs)
        if label_mode not in ['fine', 'coarse']:
            raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')
        self.label_mode = label_mode

        self.num_train_samples = 50000
        self.num_test_samples = 10000
        self.height = 32
        self.width = 32
        self.channel = 3
        self.num_class = 100
        self.transpose = (0, 2, 3, 1)

    @property
    def train_df(self):
        mode = 'train'
        shape = (self.num_train_samples, self.height * self.width * self.channel)
        rdd = self.sc.parallelize(zip(*self.load_data(mode, shape)))
        return self.rdd2df(rdd)

    @property
    def test_df(self):
        mode = 'test'
        shape = (self.num_test_samples, self.height * self.width * self.channel)
        rdd = self.sc.parallelize(zip(*self.load_data(mode, shape)))
        return self.rdd2df(rdd)

    def load_data(self, mode, shape):
        fpath = os.path.join(self.path, mode)
        x, y = load_batch(fpath, label_key=self.label_mode + '_labels')
        x = np.array(x)
        y = np.array(y)

        if K.image_data_format() == 'channels_last':
            x = x.transpose(self.transpose)

        if self.one_hot:
            y = self.convert_one_hot(y, self.num_class)

        if self.flat:
            x = x.reshape(shape)

        return x.tolist(), y.tolist()

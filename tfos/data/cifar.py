#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/25 8:45
:File   :cifar.py
:content:
  
"""
import os

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

from tfos.data import BaseData


class Cifar10(BaseData):
    def __init__(self, sc, data_dir, one_hot=False, flat=False):
        self.sc = sc
        self.data_dir = data_dir
        self.one_hot = one_hot
        self.flat = flat

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

    def convert_one_hot(self, y):
        y = np.array(y)
        num = len(y)
        labels = np.zeros((num, self.num_class))
        index = np.array(range(num)) * self.num_class + y
        labels.flat[index] = 1
        return labels.tolist()

    def load_train(self):

        x_train = np.empty((self.num_train_samples, self.channel, self.height, self.width), dtype='uint8')
        y_train = np.empty((self.num_train_samples,), dtype='uint8')

        for i in range(self.batch_size):
            fpath = os.path.join(self.data_dir, 'data_batch_' + str(i + 1))
            (x_train[i * self.num_batch_samples:i * self.num_batch_samples, :, :, :],
             y_train[i * self.num_batch_samples:i * self.num_batch_samples]) = load_batch(fpath)

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(*self.transpose)

        if self.one_hot:
            y_train = self.convert_one_hot(y_train)

        if self.flat:
            x_train = x_train.reshape((self.num_train_samples, self.height * self.width * self.channel))

        return x_train.tolist(), y_train

    def load_test(self):
        fpath = os.path.join(self.data_dir, 'test_batch')
        x_test, y_test = load_batch(fpath)

        if K.image_data_format() == 'channels_last':
            x_test = x_test.transpose(*self.transpose)

        if self.one_hot:
            y_test = self.convert_one_hot(y_test)

        if self.flat:
            x_test = x_test.reshape((self.num_test_samples, self.height * self.width * self.channel))

        return x_test.tolist(), y_test

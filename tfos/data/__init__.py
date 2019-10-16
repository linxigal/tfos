#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 15:42
:File   : __init__.py.py
"""

import numpy as np
import tensorflow as tf

from .read_data import DataSet

__all__ = ['DataSet', 'BaseData']

DATAINDEX = ['feature', 'label']


class BaseData(object):

    def __init__(self, sc, path, one_hot=False, flat=False):
        self.__sc = sc
        self.__path = path
        self.__one_hot = one_hot
        self.__flat = flat

    @property
    def sc(self):
        return self.__sc

    @property
    def path(self):
        return self.__path

    @property
    def one_hot(self):
        return self.__one_hot

    @property
    def flat(self):
        return self.__flat

    @property
    def train_df(self):
        raise NotImplementedError

    @property
    def test_df(self):
        raise NotImplementedError

    @staticmethod
    def rdd2df(rdd):
        return rdd.toDF(DATAINDEX)

    @staticmethod
    def convert_one_hot(y, num_class):
        if len(y.shape) == 0:
            label = np.zeros(num_class)
            label[y] = 1
        else:
            num = len(y)
            label = np.zeros((num, num_class))
            index = np.array(range(num)) * num_class + y
            label.flat[index] = 1
        return label

    @staticmethod
    def convert_conv(row):
        x = np.reshape(row[0], (28, 28, 1))
        return x, row[1]

    @staticmethod
    def convert_flatten(row):
        x = np.reshape(row[0], (784,))
        return x, row[1]

    @staticmethod
    def convert_one(row):
        y = row[1]
        if not isinstance(row[1], np.ndarray):
            y = np.zeros([10])
            y[row[1]] = 1
        return row[0], y

    @staticmethod
    def to_string(row):
        x, y = row
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(y, np.ndarray):
            y = y.tolist()
        return x, y

    @staticmethod
    def from_csv(s):
        return np.array([float(x) for x in s.split(',') if len(s) > 0])

    @staticmethod
    def tfr2sample(byte_str):
        example = tf.train.Example()
        example.ParseFromString(byte_str)
        features = example.features.feature
        image = np.array(features['image'].int64_list.value)
        label = np.array(features['label'].int64_list.value)
        return image, label

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 15:42
:File   : __init__.py.py
"""

import numpy as np

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
        y = np.array(y)
        if len(y.shape) == 0:
            label = np.zeros(num_class)
            label[y] = 1
        else:
            num = len(y)
            label = np.zeros((num, num_class))
            index = np.array(range(num)) * num_class + y
            label.flat[index] = 1
        return label.tolist()

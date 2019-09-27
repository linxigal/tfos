#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/10 15:42
:File   : __init__.py.py
"""
from .read_data import DataSet

__all__ = ['DataSet', 'BaseData']

DATAINDEX = ['feature', 'label']


class BaseData(object):

    @property
    def train_df(self):
        raise NotImplementedError

    @property
    def test_df(self):
        raise NotImplementedError

    @staticmethod
    def rdd2df(rdd):
        return rdd.toDF(DATAINDEX)

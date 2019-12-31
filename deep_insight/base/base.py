#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     : weijinlong
:Time:      : 2019/6/10 15:01
:File       : base.py
"""

import unittest
from collections import defaultdict

from deep_insight import *

GLOBAL_RDD = defaultdict(dict)
BRANCH = -1
BRANCH_1 = 1
BRANCH_2 = 2
DATA_BRANCH = 0
MODEL_BRANCH = 1
GP = {}


def inputRDD(name):
    res = GLOBAL_RDD[name].get(GP.get(name))
    return res


def outputRDD(name, rdd):
    GLOBAL_RDD[BRANCH][name] = rdd
    GP[BRANCH] = name


def reset():
    GLOBAL_RDD.clear()
    GP.clear()


class Base(object):

    def __init__(self, input_prev_layers=None,
                 input_rdd_name=None,
                 input_branch_1=None,
                 input_branch_2=None):

        if self.__class__.__name__ == 'InputLayer':
            model_branch = None
        elif input_prev_layers is None:
            model_branch = BRANCH
        else:
            model_branch = input_prev_layers

        data_branch = DATA_BRANCH if input_rdd_name is None else input_rdd_name
        branch_1 = BRANCH_1 if input_branch_1 is None else input_branch_1
        branch_2 = BRANCH_2 if input_branch_2 is None else input_branch_2

        self.params = dict(
            input_prev_layers=model_branch,
            input_branch_1=branch_1,
            input_branch_2=branch_2,
            input_rdd_name=data_branch
        )

    def run(self):
        raise NotImplementedError

    def p(self, key, value):
        self.params[key] = value

    def branch(self, branch):
        global BRANCH
        BRANCH = branch
        return self

    b = branch


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCase, self).__init__(*args, **kwargs)
        self.is_local = True

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

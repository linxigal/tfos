#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     : weijinlong
:Time:      : 2019/6/10 15:01
:File       : base.py
"""

import json
from collections import defaultdict

GLOBAL_RDD = defaultdict(dict)
BRANCH = -1
BRANCH_1 = 1
BRANCH_2 = 2
DATA_BRANCH = 0
GP = {}


def inputRDD(name):
    res = GLOBAL_RDD[name].get(GP.get(name))
    return res


def outputRDD(name, rdd):
    GLOBAL_RDD[BRANCH][name] = rdd
    GP[BRANCH] = name


def reset():
    GLOBAL_RDD.clear()


def lrn(branch=-1):
    if branch != -1:
        global BRANCH
        BRANCH = branch
    return GP.get(branch)


def print_pretty(name=None):
    res = {}
    data = GLOBAL_RDD.get(name if name else lrn())
    if data:
        res = json.loads(data.first().model_config)
    print(json.dumps(res, indent=4))


class Base(object):

    def __init__(self):
        self.params = dict(
            input_prev_layers=-2 if self.__class__.__name__ == 'InputLayer' else BRANCH,
            input_branch_1=BRANCH_1,
            input_branch_2=BRANCH_2,
            input_rdd_name=DATA_BRANCH
        )

    def run(self):
        raise NotImplementedError

    def p(self, key, value):
        self.params[key] = value

    def branch(self, branch=None, n1=None, n2=None):
        if branch is not None:
            global BRANCH
            BRANCH = self.valid_branch(branch)
        elif n1 is not None:
            global BRANCH_1
            BRANCH_1 = self.valid_branch(n1)
        elif n2 is not None:
            global BRANCH_2
            BRANCH_2 = self.valid_branch(n2)
        return self

    b = branch

    @staticmethod
    def valid_branch(branch):
        if branch <= 0:
            raise ValueError("custom model branch must positive!")
        return branch

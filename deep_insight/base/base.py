#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/10 15:01
:File       : base.py
"""

import json

GLOBAL_RDD = {}
GP = {}


def inputRDD(name):
    res = GLOBAL_RDD.get(name)
    return res


def outputRDD(name, rdd):
    if 'data' not in name:
        GP['LRN'] = name
    GLOBAL_RDD[name] = rdd
    SeqModel.seq_name_list.append(name)


def lrn():
    return GP.get("LRN")


class SeqModel(object):
    seq_name_list = []
    seq_model_list = []

    @classmethod
    def __get_last_name(cls):
        return cls.seq_name_list[-1]

    @property
    def l(self):
        return self.__get_last_name()


sm = SeqModel


def print_pretty(name=None):
    res = {}
    data = GLOBAL_RDD.get(name if name else lrn())
    if data:
        res = json.loads(data.first().model_config)
    print(json.dumps(res, indent=4))


class Base(object):

    def __init__(self):
        self.params = {}

    def run(self):
        raise NotImplementedError

    # def set_params(self):
    #     raise NotImplementedError

    def p(self, key, value):
        self.params[key] = value
        # setattr(self.params, key, value)

    # @property
    # def param(self):
    #     return self.params

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/10 15:01
:File       : base.py
"""

import json

global_params = []


def inputRDD(name):
    res = None
    if global_params:
        res = global_params[-1][-1]
    return res


def outputRDD(name, rdd):
    global_params.append((name, rdd))


def print_pretty(index=-1):
    key, res = '', '{}'

    if global_params:
        length = len(global_params)
        if index == -1:
            index = length

        if index < 0 or index > length:
            raise ValueError(f"第{index}个索引层不存在！！！")

        key, value = global_params[index - 1]
        res = json.loads(value.first()._1)
        res = json.dumps(res, indent=4)
    print(f"打印层： {key}")
    print(res)


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

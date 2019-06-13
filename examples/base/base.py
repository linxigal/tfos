#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/10 15:01
:File       : base.py
"""

import json

global_params = {}


def inputRDD(name):
    res = global_params.get(name)
    return res


def outputRDD(name, rdd):
    global_params[name] = rdd


def print_pretty(name):
    res = {}
    data = global_params.get(name)
    if data:
        res = json.loads(data.first().model_config)
    res = json.dumps(res, indent=4)
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

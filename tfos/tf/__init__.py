#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/10 14:14
:File   :__init__.py.py
:content:
  
"""

import tensorflow as tf
# from .model import export_tf_model, import_tf_model, TFMode
from .model import TFModel, TFCompile, TFComModel
from .base import TFModeMiddle

# __all__ = ['export_tf_model', 'import_tf_model', 'TFMode', 'TFCompile']
__all__ = ['TFModel', 'TFCompile', 'TFModeMiddle']


def add_collection(name, *args):
    for v in args:
        tf.add_to_collection(name, v)


def extract_params(obj, *args):
    params = {}
    for v in args:
        name = v.name.split(':')[0]
        value = vars(obj).get(name)
        if not value:
            raise ValueError("current object has no attr {}".format(name))
        params[name] = value

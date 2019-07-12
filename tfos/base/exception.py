#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/11 10:55
:File   : exception.py
"""

import traceback
from functools import wraps


class ext_exception(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                raise ValueError(traceback.format_exc() + "error: {} \n {}".format(self.name, e))

        return wrapper

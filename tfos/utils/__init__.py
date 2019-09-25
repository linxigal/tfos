# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 22:51
:File    : __init__.py.py
"""


def convert_bool(param):
    return True if param is True or param.lower() == 'true' else False

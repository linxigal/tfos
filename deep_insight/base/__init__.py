#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/10 15:00
:File       : __init__.py.py
"""

import os
from deep_insight import sqlc, sc
from deep_insight.base.base import *
from deep_insight.base.common import get_model_config, model2df, dict2df
from deep_insight.base.summary import SummaryLayer

__all__ = [
    'os', 'unittest',
    # spark global variable
    'sc', 'sqlc',
    # global method and variable
    'Base', 'TestCase', 'inputRDD', 'outputRDD', 'BRANCH', "BRANCH_1", "BRANCH_2", "DATA_BRANCH", 'MODEL_BRANCH',
    'reset',
    # format convert
    'get_model_config', 'model2df', 'dict2df',
    # model schema
    'SummaryLayer',
]

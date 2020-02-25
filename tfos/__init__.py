# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 23:41
:File    : __init__.py.py
"""

import os
from tfos.k.tfos import TFOS
from tfos.tf.tfos import TFOnSpark

# 版本号对应： 模块-大功能-小功能-bug解决
VERSION = (2, 2, 4, 0)

CURRENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(CURRENT_PATH)

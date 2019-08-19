#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/8/14 17:25
:File   : config.py
"""
from tfos.base.config import *

PARAM_MAP = dict(
    activation=valid_activations,
    loss=valid_losses,
    metrics=valid_metrics,

)

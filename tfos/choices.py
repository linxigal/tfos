# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

from tfos.base.config import *

CHOICES = dict(
    padding=['valid', 'same'],
    activation=[''] + valid_activations,
    loss=valid_losses
)
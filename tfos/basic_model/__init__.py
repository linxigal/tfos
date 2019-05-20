# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 22:46
:File    : __init__.py.py
"""

import os

CURRENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))
GITHUB = os.path.dirname(ROOT_PATH)
OUTPUT_DATA = os.path.join(ROOT_PATH, 'output_data')
TENSORBOARD = os.path.join(ROOT_PATH, 'tensorboard')

if not os.path.exists(OUTPUT_DATA):
    os.makedirs(OUTPUT_DATA)


if not os.path.exists(TENSORBOARD):
    os.makedirs(TENSORBOARD)
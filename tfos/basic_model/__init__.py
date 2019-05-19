# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 19/05/2019 22:46
:File    : __init__.py.py
"""

import os

CURRENT_PATH = os.getcwd()

ROOT_PATH = os.path.dirname(CURRENT_PATH)

GITHUB = os.path.dirname(ROOT_PATH)

OUTPUT_DATA = os.path.join(ROOT_PATH, 'output_data')

if not os.path.exists(OUTPUT_DATA):
    os.makedirs(OUTPUT_DATA)

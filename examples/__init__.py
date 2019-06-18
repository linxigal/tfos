#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
:Author     :weijinlong
:Time: 2019/5/22 16:51
:File       : __init__.py.py
"""

import os

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

CURRENT_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(CURRENT_PATH)
GITHUB = os.path.dirname(ROOT_PATH)
OUTPUT_DATA = os.path.join(ROOT_PATH, 'output_data')

if not os.path.exists(OUTPUT_DATA):
    os.makedirs(OUTPUT_DATA)

sc = SparkContext(conf=SparkConf().setAppName('tfos').setMaster('local[2]'))
sqlc = SQLContext(sc)

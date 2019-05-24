#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 10:01
:File       : generator.py
"""

import numpy as np


def rdd_generator(tf_feed):
    while not tf_feed.should_stop():
        batch = tf_feed.next_batch(1)
        if len(batch) == 0:
            return
        row = batch[0]
        x = np.array(row[0]).astype(np.float32) / 255.0
        y = np.array(row[1]).astype(np.int64)
        yield (x, y)

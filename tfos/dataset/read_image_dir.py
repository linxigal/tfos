#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/28 16:29
:File       : read_image_dir.py
"""

import tensorflow as tf


def read_image_dir(image_dir: str, dims: list) -> tuple:
    """

    :param image_dir:
    :param dims:
    :return: tuple (x, y)
    """


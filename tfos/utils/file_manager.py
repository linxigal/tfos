#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/7/31 17:25
:File   : file_manager.py
"""

import hdfs
import tensorflow as tf


class HDFSClient(object):
    def __init__(self):
        self.client = None,
        self.path = None

    def _client(self, path):
        _, addr, file_path = path.split(':', 2)
        self.client = hdfs.Client('http:' + addr + ':50070')
        self.path = '/' + file_path.split('/', 1)[-1]

    def makedirs(self, path):
        self._client(path)
        self.client.makedirs(self.path)
        return self.path


def makedirs(path):
    if not tf.gfile.Exists(path):
    #     if 'hdfs' in path:
    # HDFSClient().makedirs(path)
    # else:
        tf.gfile.MakeDirs(path)

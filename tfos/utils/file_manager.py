#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author : weijinlong
:Time:  : 2019/7/31 17:25
:File   : file_manager.py
"""

import unittest
import hdfs
import tensorflow as tf


class HDFSOP(object):
    @staticmethod
    def __client(path):
        _, addr, file_path = path.split(':', 2)
        client = hdfs.Client('http:' + addr + ':50070')
        path = '/' + file_path.split('/', 1)[-1]
        return client, path

    @classmethod
    def makedirs(cls, path):
        client, path = cls.__client(path)
        client.makedirs(path)

    @classmethod
    def write(cls, path, overwrite=False):
        client, path = cls.__client(path)
        return client.write(path, overwrite=overwrite)


def makedirs(path):
    if not tf.gfile.Exists(path):
        #     if 'hdfs' in path:
        # HDFSClient().makedirs(path)
        # else:
        tf.io.gfile.makedirs(path)


class TestHDFSOP(unittest.TestCase):

    @unittest.skip("")
    def test_make_dir(self):
        pass

    @unittest.skip("")
    def test_upload_file(self):
        pass


if __name__ == '__main__':
    unittest.main()

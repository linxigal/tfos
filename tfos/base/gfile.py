#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/17 15:43
:File   :gfile.py
:content:
  
"""
import json
import os
import unittest

import numpy as np
import tensorflow as tf

from tfos.base import ROUND_NUM

OUTPUT_FORMAT = ['json', 'csv']


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32)):
            return round(float(obj), ROUND_NUM)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ModelDir(object):
    """模型目录和文件的操作"""

    def __init__(self, model_dir, result_pattern):
        self.model_dir = model_dir
        self.result_pattern = result_pattern
        self.save_dir = os.path.join(self.model_dir, 'save_model')
        self.result_dir = os.path.join(self.model_dir, 'results')
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoint')
        self.log_dir = os.path.join(self.model_dir, 'tensorboard')
        self.dirs = [self.save_dir, self.result_dir, self.checkpoint_dir, self.log_dir]

    def create_model_dir(self):
        for path in self.dirs:
            if not tf.gfile.Exists(path):
                tf.gfile.MkDir(path)

    def delete_model_dir(self):
        for path in self.dirs:
            if tf.gfile.Exists(path):
                tf.gfile.DeleteRecursively(path)

    def rebuild_model_dir(self):
        self.delete_model_dir()
        self.create_model_dir()
        return self

    def delete_result_file(self):
        pattern_path = os.path.join(self.result_dir, self.result_pattern)
        for file in tf.gfile.Glob(pattern_path):
            if tf.gfile.Exists(file):
                tf.gfile.Remove(file)

    def read_result(self, is_str=False):
        results = []
        pattern_path = os.path.join(self.result_dir, self.result_pattern)
        for path in tf.gfile.Glob(pattern_path):
            with tf.gfile.FastGFile(path, 'r') as f:
                for line in f:
                    if is_str:
                        results.append(line)
                    else:
                        results.append(json.loads(line))
        return results

    @staticmethod
    def write_result(path, results, go_on=False):
        if go_on and tf.gfile.Exists(path):
            with tf.gfile.FastGFile(path, 'a') as f:
                ModelDir.write_text(f, results)
        else:
            with tf.gfile.FastGFile(path, 'w') as f:
                ModelDir.write_text(f, results)

    @staticmethod
    def write_text(f, results):
        for result in results:
            f.write(json.dumps(result, sort_keys=True, cls=CustomEncoder) + '\n')

    @staticmethod
    def write_str(path, result, go_on=False):
        if go_on and tf.gfile.Exists(path):
            with tf.gfile.FastGFile(path, 'a') as f:
                f.write(result)
        else:
            with tf.gfile.FastGFile(path, 'w') as f:
                f.write(result)

    def to_dict(self):
        return {
            'save_dir': self.save_dir,
            'result_dir': self.result_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir
        }


class TestModelDir(unittest.TestCase):

    def setUp(self) -> None:
        self.model_dir = "hdfs://t-master:8020/data/model/mnist_mlp"

    def test_read_result_file(self):
        md = ModelDir(self.model_dir, 'train*')
        md.read_result()


if __name__ == '__main__':
    unittest.main()

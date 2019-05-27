#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : TFOS.py
"""

import os
import logging
from collections import namedtuple

from tensorflowonspark import TFCluster

from tfos.worker import Worker


class TFOSBase(object):
    input_mode = None
    cluster = None
    ARGS = namedtuple("args", ['batch_size', 'steps', 'rdma'])
    worker = Worker()

    def __init__(self, steps=1000, batch_size=1, epochs=1, rdma=0, *args, **kwargs):
        """

        :param steps:
        :param batch_size:
        :param epochs:
        :param rdma:
        :param args:
        :param kwargs:
        """
        self.steps = steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.rdma = rdma
        self.args = self.ARGS._make([self.batch_size, self.steps, self.rdma])


class TFOS(TFOSBase):
    def __init__(self, sc, main_func, app_name, cluster_size, num_ps, model_path, tensorboard=0, *args, **kwargs):
        """

        :param sc:
        :param cluster_size:
        :param num_ps:
        :param tensorboard:
        """
        self.sc = sc
        self.worker.main_func = main_func
        self.app_name = app_name
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.tensorboard = tensorboard
        path = os.path.join(model_path, self.app_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.worker.model_path = path
        super(TFOS, self).__init__(*args, **kwargs)

    def init(self, first):
        logging.error(first)
        self.set_worker_dim(first)
        self.cluster = TFCluster.run(self.sc, self.worker, self.args, self.cluster_size,
                                     self.num_ps, self.tensorboard, self.input_mode)

    def set_worker_dim(self, first):
        x_dim = len(first[0])
        self.worker.x_dim = x_dim


class TFOSRdd(TFOS):
    input_mode = TFCluster.InputMode.SPARK
    rdd = None

    def train(self, rdd):
        self.worker.mode = 'train'
        self.init(rdd.first())
        self.cluster.train(rdd)

    def inference(self, rdd):
        self.worker.mode = 'inference'
        self.init(rdd.first())
        result_rdd = self.cluster.inference(rdd)
        logging.error(result_rdd.take(10))
        result_rdd.saveAsTextFile(self.worker.get_path('inference'))

    def predict(self, rdd):
        self.worker.mode = 'prediction'
        self.init(rdd.first())
        result_rdd = self.cluster.inference(rdd)
        result_rdd.saveAsTextFile(self.worker.get_path('prediction'))
        logging.error(result_rdd.take(10))


class TFOSLocal(TFOS):
    input_mode = TFCluster.InputMode.TENSORFLOW

    def read_data(self, filepath, format='csv'):
        pass

    def train(self):
        self.worker.mode = 'train'
        self.init()

    def inference(self):
        self.worker.mode = 'inference'
        # self.init()

    def predict(self):
        self.worker.mode = 'prediction'
        # self.init()

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : TFOS.py
"""

import os
from collections import namedtuple

from tensorflowonspark import TFCluster

from tfos.graph import Worker


class TFOSBase(object):
    input_mode = None
    cluster = None
    ARGS = namedtuple("args", ['batch_size', 'steps', 'rdma'])
    worker = Worker()

    def __init__(self, steps=1000, batch_size=1, epochs=1, rdma=0, *args, **kwargs):
        """

        :param rdma:
        """
        self.steps = steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.rdma = rdma
        self.args = self.ARGS._make([self.batch_size, self.steps, self.rdma])

    def init_tfos(self, main_func, rdd):
        self.worker.main_func = main_func
        self.rdd = rdd
        self.cluster = TFCluster.run(self.sc, self.worker, self.args, self.cluster_size,
                                     self.num_ps, self.tensorboard, self.input_mode)


class TFOS(TFOSBase):
    graph = None

    def __init__(self, sc, app_name, cluster_size, num_ps, model_path, tensorboard=0, *args, **kwargs):
        """

        :param sc:
        :param cluster_size:
        :param num_ps:
        :param tensorboard:
        """
        self.sc = sc
        self.app_name = app_name
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.tensorboard = tensorboard
        path = os.path.join(model_path, self.app_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.worker.model_path = path
        super(TFOS, self).__init__(*args, **kwargs)

    def train(self):
        self.mode = 'train'

    def inference(self, inf_path):
        self.mode = 'inference'

    def predict(self, pred_path):
        self.mode = 'predict'


class TFOSRdd(TFOS):
    input_mode = TFCluster.InputMode.SPARK
    rdd = None

    def train(self):
        # self.worker.mode = 'train'
        self.cluster.train(self.rdd)
    #
    # def start(self, main_func, rdd):
    #     self.worker.main_func = main_func
    #     cluster = self.init_cluster()
    #     cluster.train(rdd)


class TFOSLocal(TFOS):
    input_mode = TFCluster.InputMode.TENSORFLOW

    def start(self, main_func, filepath, fmt):
        self.worker.main_func = main_func
        pass

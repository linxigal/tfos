#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : TFOS.py
"""

import os
import logging
from datetime import datetime
import tensorflow as tf
from collections import namedtuple
from tensorflowonspark import TFCluster


class TFOSBase(object):
    input_mode = None
    cluster = None
    iterator = None
    ARGS = namedtuple("args", ['batch_size', 'steps', 'model_path', 'rdma'])

    def __init__(self, model_path,  steps=1000, batch_size=1, epochs=1, rdma=0):
        """

        :param rdma:
        """
        self.model_path = model_path
        self.steps = steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.rdma = rdma
        self.args = self.ARGS._make([self.batch_size, self.steps, model_path, self.rdma])

    def define_model_path(self):
        self.save_path = os.path.join(self.model_path, 'model')
        self.inf_path = os.path.join(self.model_path, 'inf')
        self.pred_path = os.path.join(self.model_path, 'pred')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.inf_path):
            os.makedirs(self.inf_path)
        if not os.path.exists(self.pred_path):
            os.makedirs(self.pred_path)

    @staticmethod
    def worker_fun(task_index, cluster):
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

    @staticmethod
    def map_fun(args, ctx):
        job_name = ctx.job_name
        task_index = ctx.task_index

        # Get TF cluster and server instances
        cluster, server = ctx.start_cluster_server(1, args.rdma)

        # Create generator for Spark data feed
        tf_feed = ctx.get_data_feed(args.mode == 'train')


class TFOS(TFOSBase):
    graph = None

    def __init__(self, sc, cluster_size, num_ps, tensorboard=0, *args, **kwargs):
        """

        :param sc:
        :param cluster_size:
        :param num_ps:
        :param tensorboard:
        """
        self.sc = sc
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.tensorboard = tensorboard
        super(TFOS, self).__init__(*args, **kwargs)

    def init_cluster(self):
        cluster = TFCluster.run(self.sc,
                                self.map_fun, self.args, self.cluster_size, self.num_ps, self.tensorboard, self.input_mode)

    def build_model(self, model, optimizer):
        print(self.input_mode)

    def train(self):
        self.mode = 'train'
        self.init_cluster()

    def inference(self, inf_path):
        self.mode = 'inference'

    def predict(self, pred_path):
        self.mode = 'predict'


class TFOSRdd(TFOS):
    input_mode = TFCluster.InputMode.SPARK

    def __init__(self, rdd, *args, **kwargs):
        """learning data from spark rdd

        :param rdd:
        """
        self.rdd = rdd
        super(TFOSRdd, self).__init__(*args, **kwargs)

    def init_data(self):
        pass


class TFOSLocal(TFOS):
    input_mode = TFCluster.InputMode.TENSORFLOW

    def __init__(self,  filepath, fmt, *args, **kwargs):
        """learning data from file source, read data via tensorflow

        :param filepath: filename path
        :param fmt: file format, example for csv, txt, bin, tfr, etc
        """
        self.filepath = filepath
        self.fmt = fmt
        super(TFOSLocal, self).__init__(*args, **kwargs)

    def init_data(self):
        pass

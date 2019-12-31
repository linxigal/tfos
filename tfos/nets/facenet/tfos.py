#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/20 10:09
:File   :facenet_tfos.py
:content:
  
"""

import math

from tensorflowonspark import TFCluster

from tfos.base import ext_exception
from tfos.base.gfile import ModelDir
from .worker import FaceNetSoftMaxWorker


class TFOSFaceNetSoftMax(object):
    def __init__(self, sc, sqlc=None, cluster_size=2, num_ps=1, input_mode=TFCluster.InputMode.SPARK):
        self.sc = sc
        self.sqlc = sqlc
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.input_mode = input_mode
        self.cluster = None
        self.tf_args = {}

    @property
    def num_workers(self):
        num_workers = self.cluster_size - self.num_ps
        assert num_workers > 0, "cluster_size, num_ps must be positive, and cluster_size > num_ps"
        return num_workers

    @ext_exception("mtcnn process")
    def train(self, data_rdd, model_rdd, batch_size, epochs, model_dir, go_on=False, *args, **kwargs):
        n_classes = data_rdd.count()
        n_samples = data_rdd.count()
        # steps_per_epoch = n_samples // batch_size // self.num_workers + 1
        steps_per_epoch = math.ceil(n_samples / batch_size / self.num_workers)
        md = ModelDir(model_dir, 'train*')
        if go_on:
            md.create_model_dir()
        else:
            md = md.rebuild_model_dir()
        worker = FaceNetSoftMaxWorker(model_rdd=model_rdd,
                                      n_classes=n_classes,
                                      go_on=go_on,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      **md.to_dict())
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=epochs, feed_timeout=60000)
        cluster.shutdown()
        results = md.read_result()
        if results:
            return self.sqlc.createDataFrame(results)

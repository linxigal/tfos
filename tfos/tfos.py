#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : tfos.py
"""

from tensorflowonspark import TFCluster

from tfos.base import ext_exception
from .worker import TrainWorker, InferenceWorker


class TFOS(object):
    def __init__(self, sc, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK):
        self.sc = sc
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.input_mode = input_mode

    @ext_exception("train model")
    def train(self, data_rdd, model_rdd, batch_size, epochs, model_dir):
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        n_samples = data_rdd.count()
        steps_per_epoch = n_samples // batch_size
        # steps_per_epoch = 1
        worker = TrainWorker(model_rdd, batch_size, epochs, steps_per_epoch, model_dir)
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.train(data_rdd.rdd)

    @ext_exception('inference model')
    def inference(self, data_rdd, model_rdd, model_dir):
        n_samples = data_rdd.count()
        steps_per_epoch = n_samples
        # steps_per_epoch = 5
        worker = InferenceWorker(model_rdd, steps_per_epoch=steps_per_epoch, model_dir=model_dir)
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        output_rdd = cluster.inference(data_rdd.rdd)
        return output_rdd.toDF()

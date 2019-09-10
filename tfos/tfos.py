#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : tfos.py
"""

import os
from tensorflowonspark import TFCluster

# from tfos import cluster as TFCluster
from tfos.base import ext_exception
from .worker import TrainWorker, EvaluateWorker, PredictWorker
from tfos.base.gfile import ModeDir


class TFOS(object):
    def __init__(self, sc, sqlc, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK):
        self.sc = sc
        self.sqlc = sqlc
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.input_mode = input_mode
        self.cluster = None

    @ext_exception("train model")
    def train(self, data_rdd, model_rdd, batch_size, epochs, model_dir):
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        n_samples = data_rdd.count()
        steps_per_epoch = n_samples // batch_size
        md = ModeDir(model_dir, 'train*').rebuild_model_dir()
        worker = TrainWorker(model_rdd,
                             batch_size=batch_size,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             **md.to_dict())
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.train(data_rdd.rdd)
        cluster.shutdown()
        return self.sqlc.createDataFrame(md.read_result_file())

    @ext_exception('evaluate model')
    def evaluate(self, data_rdd, steps, model_dir):
        md = ModeDir(model_dir, 'evaluate*')
        steps_per_epoch = data_rdd.count() if steps <= 0 else steps
        worker = EvaluateWorker(steps_per_epoch=steps_per_epoch, **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.train(data_rdd.rdd)
        cluster.shutdown()
        return self.sqlc.createDataFrame(md.read_result_file())

    @ext_exception('predict model')
    def predict(self, data_rdd, steps, model_dir):
        md = ModeDir(model_dir, 'predict*')
        steps_per_epoch = data_rdd.count() if steps <= 0 else steps
        worker = PredictWorker(steps_per_epoch=steps_per_epoch,
                               **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.train(data_rdd.rdd)
        cluster.shutdown()
        return self.sqlc.createDataFrame(md.read_result_file())

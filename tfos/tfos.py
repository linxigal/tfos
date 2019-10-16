#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : tfos.py
"""

from tensorflowonspark import TFCluster

from tfos.base import ext_exception
from tfos.base.gfile import ModelDir
from .worker import TrainWorker, EvaluateWorker, PredictWorker


class TFOS(object):
    def __init__(self, sc, sqlc, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK):
        self.sc = sc
        self.sqlc = sqlc
        self.cluster_size = cluster_size
        self.num_ps = num_ps
        self.input_mode = input_mode
        self.cluster = None

    @property
    def num_workers(self):
        num_workers = self.cluster_size - self.num_ps
        assert num_workers > 0, "cluster_size, num_ps must be positive, and cluster_size > num_ps"
        return num_workers

    @ext_exception("train model")
    def train(self, data_rdd, model_rdd, batch_size, epochs, model_dir, go_on=False):
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        n_samples = data_rdd.count()
        steps_per_epoch = n_samples * epochs // batch_size // self.num_workers
        md = ModelDir(model_dir, 'train*')
        if go_on:
            md.create_model_dir()
        else:
            md = md.rebuild_model_dir()
        worker = TrainWorker(model_rdd,
                             go_on=go_on,
                             batch_size=batch_size,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             **md.to_dict())
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=epochs)
        cluster.shutdown()
        results = md.read_result()
        if results:
            return self.sqlc.createDataFrame(results)

    @ext_exception('evaluate model')
    def evaluate(self, data_rdd, steps, model_dir):
        md = ModelDir(model_dir, 'evaluate*')
        steps_per_epoch = data_rdd.count() if steps <= 0 else steps
        steps_per_epoch = steps_per_epoch // self.num_workers
        worker = EvaluateWorker(steps_per_epoch=steps_per_epoch, **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=1)
        cluster.shutdown()
        results = md.read_result()
        if results:
            return self.sqlc.createDataFrame(results)

    @ext_exception('predict model')
    def predict(self, data_rdd, steps, model_dir, output_prob=False):
        md = ModelDir(model_dir, 'predict*')
        steps_per_epoch = data_rdd.count() if steps <= 0 else steps
        steps_per_epoch = steps_per_epoch // self.num_workers
        worker = PredictWorker(steps_per_epoch=steps_per_epoch,
                               output_prob=output_prob,
                               **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=1)
        cluster.shutdown()
        results = md.read_result()
        if results:
            return self.sqlc.createDataFrame(results)

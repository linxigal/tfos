#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : tfos.py
"""
import math

from tensorflowonspark import TFCluster

from tfos.base import ext_exception
from tfos.base.gfile import ModelDir
from .worker import TrainWorker, EvaluateWorker, PredictWorker, RecurrentPredictWorker


class TFOS(object):
    def __init__(self, sc, sqlc, cluster_size=2, num_ps=1, input_mode=TFCluster.InputMode.SPARK):
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

    @ext_exception("train model")
    def train(self, data_rdd, model_rdd, batch_size, epochs, model_dir, go_on=False):
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        n_samples = data_rdd.count()
        # steps_per_epoch = n_samples // batch_size // self.num_workers + 1
        steps_per_epoch = math.ceil(n_samples / batch_size / self.num_workers)
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
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=epochs, feed_timeout=60000)
        cluster.shutdown()
        results = md.read_result()
        return self.sqlc.createDataFrame(results)

    @ext_exception('evaluate model')
    def evaluate(self, data_rdd, steps, model_dir):
        md = ModelDir(model_dir, 'evaluate*')
        steps_per_epoch = data_rdd.count() if steps <= 0 else steps
        steps_per_epoch = math.ceil(steps_per_epoch / self.num_workers)
        worker = EvaluateWorker(steps_per_epoch=steps_per_epoch, **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=1)
        cluster.shutdown()
        results = md.read_result()
        return self.sqlc.createDataFrame(results)

    @ext_exception('predict model')
    def predict(self, data_rdd, steps, model_dir, output_prob=False):
        md = ModelDir(model_dir, 'predict*')
        steps_per_epoch = data_rdd.count() if steps <= 0 else steps
        steps_per_epoch = math.ceil(steps_per_epoch / self.num_workers)
        worker = PredictWorker(steps_per_epoch=steps_per_epoch,
                               output_prob=output_prob,
                               **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=1, feed_timeout=6000)
        cluster.shutdown()
        results = md.read_result()
        return self.sqlc.createDataFrame(results)

    @ext_exception('recurrent predict model')
    def recurrent_predict(self, data_rdd, units, steps, feature_type, model_dir):
        md = ModelDir(model_dir, 'recurrent_predict*')
        worker = RecurrentPredictWorker(units=units,
                                        steps=steps,
                                        feature_type=feature_type,
                                        **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=1, feed_timeout=6000)
        cluster.shutdown()
        results = md.read_result(True)
        return self.sqlc.createDataFrame([{"result": result} for result in results])

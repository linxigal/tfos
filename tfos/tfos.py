#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : tfos.py
"""
import math

import tensorflow as tf
from tensorflowonspark import TFCluster

from tfos.base import ext_exception
from tfos.base.gfile import ModelDir
from .nets.yolov3.train import YOLOV3ModelTrainWorker, YOLOV3TinyModelTrainWorker
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
        cluster.train(data_rdd.rdd, num_epochs=epochs)
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
        cluster.train(data_rdd.rdd, num_epochs=1)
        cluster.shutdown()
        results = md.read_result()
        return self.sqlc.createDataFrame(results)

    @ext_exception('recurrent predict model')
    def recurrent_predict(self, data_rdd, units, steps, model_dir):
        md = ModelDir(model_dir, 'recurrent_predict*')
        worker = RecurrentPredictWorker(units=units,
                                        steps=steps,
                                        **md.to_dict())
        md.delete_result_file()
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd.rdd, num_epochs=1)
        cluster.shutdown()
        results = md.read_result(True)
        return self.sqlc.createDataFrame([{"result": result} for result in results])

    @ext_exception('yolov3 train model')
    def yolov3_train(self, model_rdd, batch_size, epochs, classes_path, anchors_path,
                     train_path, val_path, image_size, model_dir, weights_path=None, freeze_body=2, go_on=False):
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert tf.io.gfile.exists(train_path), "train dataset path not exists!"
        data_rdd = self.sc.textFile(train_path)
        n_samples = data_rdd.count()
        print(n_samples)
        steps_per_epoch = math.ceil(n_samples / batch_size / self.num_workers)
        md = ModelDir(model_dir, 'train*')
        if go_on:
            md.create_model_dir()
        else:
            md = md.rebuild_model_dir()
        worker = YOLOV3ModelTrainWorker(model_rdd,
                                        go_on=go_on,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        classes_path=classes_path,
                                        anchors_path=anchors_path,
                                        weights_path=weights_path,
                                        val_path=val_path,
                                        image_size=image_size,
                                        steps_per_epoch=steps_per_epoch,
                                        freeze_body=freeze_body,
                                        **md.to_dict())
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd, num_epochs=epochs)
        cluster.shutdown()
        results = md.read_result()
        return self.sqlc.createDataFrame(results)

    @ext_exception('yolov3 tiny train model')
    def yolov3_tiny_train(self, model_rdd, batch_size, epochs, classes_path, anchors_path, weights_path,
                          train_path, val_path, image_size, model_dir, freeze_body=2, go_on=False):
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert tf.io.gfile.exists(train_path), "train dataset path not exists!"
        data_rdd = self.sc.textFile(train_path)
        n_samples = data_rdd.count()
        steps_per_epoch = math.ceil(n_samples / batch_size / self.num_workers)
        md = ModelDir(model_dir, 'train*')
        if go_on:
            md.create_model_dir()
        else:
            md = md.rebuild_model_dir()
        worker = YOLOV3TinyModelTrainWorker(model_rdd,
                                            go_on=go_on,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            classes_path=classes_path,
                                            anchors_path=anchors_path,
                                            weights_path=weights_path,
                                            val_path=val_path,
                                            image_size=image_size,
                                            steps_per_epoch=steps_per_epoch,
                                            freeze_body=freeze_body,
                                            **md.to_dict())
        cluster = TFCluster.run(self.sc, worker, self.tf_args, self.cluster_size, self.num_ps,
                                input_mode=self.input_mode)
        cluster.train(data_rdd, num_epochs=epochs)
        cluster.shutdown()
        results = md.read_result()
        return self.sqlc.createDataFrame(results)

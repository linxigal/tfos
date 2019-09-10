# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 26/05/2019 19:00
:File    : graph.py
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflowonspark import TFNode

from tfos.base.gfile import ModeDir


class Worker(object):
    def __init__(self, batch_size=1,
                 epochs=1,
                 steps_per_epoch=1,
                 name="model",
                 save_dir=None,
                 result_dir=None,
                 checkpoint_dir=None,
                 log_dir=None):
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.task_index = None
        self.job_name = None
        self.tf_feed = None
        self.cluster = None
        self.server = None
        self.model = None
        self.tmp_dir = "/tmp"
        # self.save_dir = self.create_dir('save_model')
        # self.result_dir = self.create_dir('results')
        # self.checkpoint_dir = self.create_dir('checkpoint')
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_checkpoint_{epoch}')

    # def create_dir(self, path):
    #     path = os.path.join(self.model_dir, path)
    #     if self.__class__.__name__ == "TrainWorker":
    #         if tf.gfile.Exists(path):
    #             tf.gfile.DeleteRecursively(path)
    #         tf.gfile.MkDir(path)
    #     return path

    @property
    def model_name(self, suffix='.h5'):
        return self.name + suffix

    @property
    def model_tmp_path(self):
        return os.path.join(self.tmp_dir, self.model_name)

    @property
    def model_save_path(self):
        return os.path.join(self.save_dir, self.model_name)

    def generate_rdd_data(self):
        while not self.tf_feed.should_stop():
            batches = self.tf_feed.next_batch(self.batch_size)
            inputs = []
            labels = []
            for row in batches:
                inputs.append(row.features)
                labels.append(row.label)
            inputs = np.array(inputs).astype('float32')
            labels = np.array(labels).astype('float32')
            yield inputs, labels

    def build_model(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError

    def save_model(self):
        self.model.save(self.model_tmp_path)
        tf.gfile.Copy(self.model_tmp_path, self.model_save_path, True)

    def load_model(self):
        tf.gfile.Copy(self.model_save_path, self.model_tmp_path, True)
        self.model = load_model(self.model_tmp_path)

    def __call__(self, args, ctx):
        self.task_index = ctx.task_index
        self.job_name = ctx.job_name
        self.cluster, self.server = TFNode.start_cluster_server(ctx)
        self.tf_feed = TFNode.DataFeed(ctx.mgr)
        if ctx.job_name == "ps":
            self.server.join()
        elif ctx.job_name == "worker":
            self.build_model()
            self.execute()


class TrainWorker(Worker):
    """训练"""

    def __init__(self, model_rdd, *args, **kwargs):
        super(TrainWorker, self).__init__(*args, **kwargs)
        self.is_sequence = model_rdd.first().is_sequence
        self.model_config = json.loads(model_rdd.first().model_config)
        self.compile_config = json.loads(model_rdd.first().compile_config)

    def get_results(self, his):
        results = []
        length = 0
        for key, values in his.history.items():
            length = len(values)
            results.append(zip([key] * len(values), values))
        results.append(zip(['_tast_index'] * length, [self.task_index] * length))
        results.append(zip(['_epoch'] * length, his.epoch))
        return [dict(v) for v in zip(*results)]

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            if self.is_sequence:
                model = Sequential.from_config(self.model_config)
            else:
                model = Model.from_config(self.model_config)
            model.compile(**self.compile_config)
            model.summary()
            self.model = model

    def execute(self):
        result_file = os.path.join(self.result_dir, "train_result_{}.txt".format(self.task_index))
        with tf.Session(self.server.target) as sess:
            sess.run(tf.global_variables_initializer())

            tb_callback = TensorBoard(log_dir=self.log_dir, write_grads=True, write_images=True)
            ckpt_callback = ModelCheckpoint(self.checkpoint_file, monitor='loss', save_weights_only=True,
                                            save_best_only=True)

            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            callbacks = [tb_callback, ckpt_callback] if self.task_index == 0 else None

            # train on data read from a generator which is producing data from a Spark RDD
            his = self.model.fit_generator(generator=self.generate_rdd_data(),
                                           steps_per_epoch=self.steps_per_epoch,
                                           epochs=self.epochs,
                                           callbacks=callbacks
                                           )
            if self.task_index == 0:
                self.save_model()

            ModeDir.write_result(result_file, self.get_results(his))
            self.tf_feed.terminate()


class EvaluateWorker(Worker):
    """评估"""

    def get_results(self, his):
        if isinstance(his, list):
            his = [float(v) for v in his]
            result = zip(["_task_index"] + self.model.metrics_names, [self.task_index] + his)
        else:
            result = zip(['_task_index', 'loss'], [self.task_index, float(his)])
        return [dict(result)]

    def build_model(self):
        self.load_model()

    def execute(self):
        result_file = os.path.join(self.result_dir, "evaluate_result_{}.txt".format(self.task_index))
        with tf.Session(self.server.target) as sess:
            sess.run(tf.global_variables_initializer())
            his = self.model.evaluate_generator(generator=self.generate_rdd_data(),
                                                steps=self.steps_per_epoch)
            ModeDir.write_result(result_file, self.get_results(his))
            self.tf_feed.terminate()


class PredictWorker(Worker):
    """预测"""

    def get_results(self, his):
        results = []
        for pred in his:
            results.append({
                '_task_index': self.task_index,
                'label': int(np.argmax(pred))
            })
        return results

    def build_model(self):
        self.load_model()

    def execute(self):
        result_file = os.path.join(self.result_dir, "predict_result_{}.txt".format(self.task_index))
        with tf.Session(self.server.target) as sess:
            sess.run(tf.global_variables_initializer())
            his = self.model.predict_generator(self.generate_rdd_data(),
                                               steps=self.steps_per_epoch)
            ModeDir.write_result(result_file, self.get_results(his))
            self.tf_feed.terminate()

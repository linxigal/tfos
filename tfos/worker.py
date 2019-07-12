# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 26/05/2019 19:00
:File    : graph.py
"""

import os
import json

import numpy as np
import tensorflow as tf
from pyspark.sql import Row
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflowonspark import TFNode


class Worker(object):
    def __init__(self, model_rdd, batch_size=1, epochs=1, steps_per_epoch=1, model_dir=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_dir = model_dir
        self.task_index = None
        self.job_name = None
        self.tf_feed = None
        self.cluster = None
        self.server = None
        self.checkpoint_path = os.path.join(self.model_dir, "checkpoint")
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        # self.checkpoint_file = os.path.join(self.checkpoint_path, 'model_checkpoint.h5')
        self.checkpoint_file = os.path.join(self.checkpoint_path, 'model_checkpoint_{epoch}')
        self.tensorboard_dir = os.path.join(self.model_dir, "tensorboard")
        self.save_dir = os.path.join(self.model_dir, "save_model")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_model_file = os.path.join(self.save_dir, 'model.h5')

        self.model_config = json.loads(model_rdd.first().model_config)
        self.compile_config = json.loads(model_rdd.first().compile_config)

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

    def execute(self):
        raise NotImplementedError

    def __call__(self, args, ctx):
        self.task_index = ctx.task_index
        self.job_name = ctx.job_name
        self.cluster, self.server = TFNode.start_cluster_server(ctx)
        self.tf_feed = TFNode.DataFeed(ctx.mgr)

        if ctx.job_name == "ps":
            self.server.join()
        elif ctx.job_name == "worker":
            self.execute()


class TrainWorker(Worker):
    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            model = Sequential.from_config(self.model_config)

            if os.path.exists(self.checkpoint_file) and self.task_index == 0:
                model.load_weights(self.checkpoint_file)

            model.compile(**self.compile_config)
            model.summary()
        return model

    def execute(self):
        model = self.build_model()
        with tf.Session(self.server.target) as sess:
            sess.run(tf.global_variables_initializer())

            tb_callback = TensorBoard(log_dir=self.tensorboard_dir, write_grads=True, write_images=True)
            ckpt_callback = ModelCheckpoint(self.checkpoint_file, monitor='loss', save_weights_only=True,
                                            save_best_only=True)

            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            callbacks = [tb_callback, ckpt_callback] if self.task_index == 0 else None

            # train on data read from a generator which is producing data from a Spark RDD
            model.fit_generator(generator=self.generate_rdd_data(),
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=self.epochs,
                                callbacks=callbacks
                                )
            if self.save_model_file and self.task_index == 0:
                model.save(self.save_model_file)
            self.tf_feed.terminate()


class InferenceWorker(Worker):
    def execute(self):
        with tf.Session(self.server.target) as sess:
            if os.path.exists(self.save_model_file) and self.task_index == 0:
                model = load_model(self.save_model_file)
            else:
                raise ValueError("model train result file is not exists!!!")

            for i in range(self.steps_per_epoch):
                x, y = next(self.generate_rdd_data())
                result = model.evaluate(x, y, self.batch_size)
                # numpy.float32 cannot convert to DataFrame
                self.tf_feed.batch_results([Row(loss=float(result[0]), acc=float(result[1]))])
                # self.tf_feed.batch_results([result])
            self.tf_feed.terminate()

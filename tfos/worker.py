# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 26/05/2019 19:00
:File    : graph.py
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.optimizers import deserialize
from tensorflowonspark import TFNode

from tfos.base import ModelType, gmt, logger
from tfos.base.gfile import ModelDir


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
        self.labels = []
        # self.tmp_dir = "/tmp"
        self.tmp_dir = "/tmp/tfos/{}".format(int(time.time() * 1000))
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_checkpoint_{epoch}')

    def create_tmp_dir(self):
        # tf.io.gfile.mkdir(self.tmp_dir)
        tf.io.gfile.makedirs(self.tmp_dir)

    def delete_tmp_dir(self):
        if tf.io.gfile.exists(self.tmp_dir):
            tf.io.gfile.rmtree(self.tmp_dir)

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
            if not batches:
                raise StopIteration()
            for row in batches:
                inputs.append(row.feature)
                labels.append(row.label)
                self.labels.append(np.argmax(row.label))
            inputs = np.array(inputs).astype('float32')
            labels = np.array(labels).astype('float32')
            yield inputs, labels

    def build_model(self):
        pass

    def execute(self):
        raise NotImplementedError

    def save_model(self):
        if self.task_index == 0:
            self.model.save(self.model_tmp_path)
            tf.io.gfile.copy(self.model_tmp_path, self.model_save_path, True)
            tf.io.gfile.rmtree(self.tmp_dir)

    def load_model(self):
        tf.io.gfile.copy(self.model_save_path, self.model_tmp_path, True)
        K.set_learning_phase(False)
        self.model = load_model(self.model_tmp_path)

    def __call__(self, args, ctx):
        self.task_index = ctx.task_index
        self.job_name = ctx.job_name
        self.cluster, self.server = TFNode.start_cluster_server(ctx)
        self.tf_feed = TFNode.DataFeed(ctx.mgr)
        if ctx.job_name == "ps":
            self.server.join()
        elif ctx.job_name == "worker":
            self.create_tmp_dir()
            self.build_model()
            self.execute()
            self.delete_tmp_dir()


class TrainWorker(Worker):
    """训练"""

    def __init__(self, model_rdd, go_on, *args, **kwargs):
        super(TrainWorker, self).__init__(*args, **kwargs)
        self.model_config = json.loads(model_rdd.first().model_config)
        self.compile_config = json.loads(model_rdd.first().compile_config)
        self.go_on = go_on
        self.initial_epoch = 0

    def parse_optimizer(self):
        optimizer = self.compile_config.get('optimizer')
        if optimizer and isinstance(optimizer, dict):
            self.compile_config['optimizer'] = deserialize(optimizer)

    def get_results(self, his):
        results = []
        length = 0
        for key, values in his.history.items():
            length = len(values)
            results.append(zip([key] * len(values), values))
        results.append(zip(['_task_index'] * length, [self.task_index] * length))
        results.append(zip(['_epoch'] * length, his.epoch))
        return [dict(v) for v in zip(*results)]

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            model_type = gmt(self.model_config)
            if model_type == ModelType.SEQUENCE:
                model = Sequential.from_config(self.model_config)
            elif model_type == ModelType.NETWORK:
                model = Model.from_config(self.model_config)
            else:
                raise ValueError("{}, unknown model type!!!".format(model_type))
            self.parse_optimizer()
            model.compile(**self.compile_config)
            self.model = model

    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            K.set_learning_phase(False)
            self.initial_epoch = int(ckpt.model_checkpoint_path.split('_')[-1])
            self.model.load_weights(ckpt.model_checkpoint_path)

    def execute(self):
        result_file = os.path.join(self.result_dir, "train_result_{}.txt".format(self.task_index))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(self.server.target, config=config) as sess:
            K.set_session(sess)
            if self.go_on:
                self.restore_model()
            tb_callback = TensorBoard(log_dir=self.log_dir, write_grads=True, write_images=True)
            ckpt_callback = ModelCheckpoint(self.checkpoint_file, monitor='loss', save_weights_only=True)

            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            callbacks = [tb_callback, ckpt_callback] if self.task_index == 0 else None

            # train on data read from a generator which is producing data from a Spark RDD
            his = self.model.fit_generator(generator=self.generate_rdd_data(),
                                           steps_per_epoch=self.steps_per_epoch,
                                           epochs=self.epochs + self.initial_epoch,
                                           callbacks=callbacks,
                                           workers=0,
                                           initial_epoch=self.initial_epoch)
            self.save_model()
            ModelDir.write_result(result_file, self.get_results(his), self.go_on)
            self.tf_feed.terminate()


class EvaluateWorker(Worker):
    """评估"""

    def get_results(self, his):
        if isinstance(his, list):
            result = zip(["_task_index"] + self.model.metrics_names, [self.task_index] + his)
        else:
            result = zip(['_task_index', 'loss'], [self.task_index, his])
        return [dict(result)]

    def execute(self):
        result_file = os.path.join(self.result_dir, "evaluate_result_{}.txt".format(self.task_index))
        with tf.Session(self.server.target) as sess:
            K.set_session(sess)
            self.load_model()
            his = self.model.evaluate_generator(generator=self.generate_rdd_data(),
                                                steps=self.steps_per_epoch)
            ModelDir.write_result(result_file, self.get_results(his))
            self.tf_feed.terminate()


class PredictWorker(Worker):
    """预测"""

    def __init__(self, output_prob=False, *args, **kwargs):
        super(PredictWorker, self).__init__(*args, **kwargs)
        self.output_prob = output_prob

    def get_results(self, his):
        results = []
        length = len(his)
        results.append([('_task_index', self.task_index)] * length)
        results.append(zip(['p_label'] * length, [np.argmax(v) for v in his]))
        results.append(zip(['true'] * length, self.labels))
        if self.output_prob:
            results.extend(zip(*[[('z_{}'.format(i), v) for i, v in enumerate(value)] for value in his]))
        return [dict(v) for v in zip(*results)]

    def execute(self):
        result_file = os.path.join(self.result_dir, "predict_result_{}.txt".format(self.task_index))
        with tf.Session(self.server.target) as sess:
            K.set_session(sess)
            self.load_model()
            his = self.model.predict_generator(self.generate_rdd_data(),
                                               steps=self.steps_per_epoch)
            ModelDir.write_result(result_file, self.get_results(his))
            self.tf_feed.terminate()


class RecurrentPredictWorker(Worker):
    """循环预测"""

    def __init__(self, units, steps, feature_type, *args, **kwargs):
        super(RecurrentPredictWorker, self).__init__(*args, **kwargs)
        self.units = units
        self.steps = steps
        self.feature_type = feature_type

    def execute(self):
        result_file = os.path.join(self.result_dir, "recurrent_predict_result_{}.txt".format(self.task_index))
        with tf.Session(self.server.target) as sess:
            K.set_session(sess)
            self.load_model()
            for x, y in self.generate_rdd_data():
                x_len = x.shape[1]
                if x_len < self.units:
                    break
                x_train = np.array(x[:self.units])
                for _ in range(self.steps):
                    ys = self.model.predict(x_train, batch_size=1)
                    y_label = np.argmax(ys, 1)
                    if self.feature_type == 'one_hot':
                        shape = ys.shape
                        y_l = np.zeros(shape)
                        y_l[..., y_label] = 1
                        x_train = np.array([x_train.tolist()[0][1:] + y_l.tolist()])
                    else:
                        x_train = np.array([x_train.tolist()[0][1:] + y_label.tolist()])

                    ModelDir.write_str(result_file, str(y_label.tolist()[0]) + " ", True)
                ModelDir.write_str(result_file, "\n", True)

            self.tf_feed.terminate()

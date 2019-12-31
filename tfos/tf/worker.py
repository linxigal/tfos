#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/17 10:32
:File   :worker.py
:content:
  
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflowonspark import TFNode

from tfos.base.gfile import ModelDir
from tfos.tf.model import import_model, TFMode


class TFWorker(object):
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
        self.tmp_dir = "/tmp/tfos/tf/{}".format(int(time.time() * 1000))
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name + '_checkpoint_{epoch}')

    def create_tmp_dir(self):
        # tf.io.gfile.mkdir(self.tmp_dir)
        tf.io.gfile.makedirs(self.tmp_dir)

    def delete_tmp_dir(self):
        if tf.io.gfile.exists(self.tmp_dir):
            tf.io.gfile.rmtree(self.tmp_dir)

    @property
    def global_step(self):
        return tf.train.get_global_step()

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
        """
        保存h5文件
        :return:
        """
        if self.task_index == 0:
            self.model.save(self.model_tmp_path)
            tf.gfile.Copy(self.model_tmp_path, self.model_save_path, True)

    def load_model(self):
        """
        加载h5文件
        :return:
        """
        tf.gfile.Copy(self.model_save_path, self.model_tmp_path, True)
        K.set_learning_phase(False)
        # self.model = load_model(self.model_tmp_path)

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


class TFTrainWorker(TFWorker):
    def __init__(self, model_rdd, go_on=False, *args, **kwargs):
        super(TFTrainWorker, self).__init__(*args, **kwargs)
        self.model_rdd = model_rdd
        self.go_on = go_on
        self.initial_epoch = 0
        self.input_dict, self.output_dict = {}, {}

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            self.model = TFMode().deserialize(self.model_rdd)

    def restore_model(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            K.set_learning_phase(False)
            self.initial_epoch = int(ckpt.model_checkpoint_path.split('_')[-1])
            saver.restore(sess, self.checkpoint_dir)

    def execute(self):
        result_file = os.path.join(self.result_dir, "train_result_{}.txt".format(self.task_index))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        summary_op = tf.summary.merge_all()
        with tf.Session(self.server.target, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            if self.go_on:
                self.restore_model(sess)
            keys = ["_task_index", "_epoch"]
            for epoch in range(1, self.epochs + 1):
                for _ in range(self.steps_per_epoch - 1):
                    sess.run(self.model.fetches, self.model.feed_dict)
                res = sess.run(self.model.fetches + [summary_op], self.model.feed_dict)
                res = [self.task_index, epoch] + res[:-1]
                results = [dict(zip(keys, res))]
                ModelDir.write_result(result_file, results, True)
                summary_writer.add_summary(res[-1], global_step=sess.run(tf.train.get_or_create_global_step()))
            self.tf_feed.terminate()


class TFEvaluateWorker(TFWorker):

    def execute(self):
        pass


class TFPredictWorker(TFWorker):

    def execute(self):
        pass

#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/17 10:32
:File   :worker.py
:content:
  
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflowonspark import TFNode

from tfos.base import logger
from tfos.base.gfile import ModelDir
from tfos.tf import TFCompile


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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)

    @property
    def global_step(self):
        return tf.train.get_global_step()

    @property
    def model_name(self, suffix='.ckpt'):
        return self.name + suffix

    @property
    def model_path(self):
        return os.path.join(self.save_dir, 'model.pb')

    @property
    def model_config_path(self):
        return os.path.join(self.save_dir, 'model.config')

    def common_dict(self, epoch):
        return {
            "_task_index": self.task_index,
            "_epoch": epoch
        }

    @property
    def generate_rdd_data(self):
        while not self.tf_feed.should_stop():
            batches = self.tf_feed.next_batch(self.batch_size)
            inputs = []
            labels = []
            # if not batches:
            #     raise StopIteration()
            for row in batches:
                inputs.append(row.feature)
                labels.append(row.label)
                self.labels.append(np.argmax(row.label))
            inputs = np.array(inputs).astype('float32')
            labels = np.array(labels).astype('float32')
            return inputs, labels

    def feed_dict(self, x, y=None):
        data = self.model.feed_dict
        data[self.model.inputs['x']] = x
        if y is not None:
            data[self.model.inputs['y']] = y
        # else:
        #     data.pop(self.model.inputs['y'])
        return data

    def build_model(self):
        pass

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
            self.build_model()
            self.execute()


class TFTrainWorker(TFWorker):
    def __init__(self, model_rdd, go_on=False, *args, **kwargs):
        super(TFTrainWorker, self).__init__(*args, **kwargs)
        self.model_config = json.loads(getattr(model_rdd.first(), TFCompile.MODEL))
        self.go_on = go_on
        self.initial_epoch = 0
        self.summary_writer = None

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            self.model = TFCompile.from_json(self.model_config)
            self.model.valid_model()

    def restore_checkpoint(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # K.set_learning_phase(False)
            self.initial_epoch = int(ckpt.model_checkpoint_path.split('-')[-1])
            saver.restore(sess, ckpt.model_checkpoint_path)

    def save_checkpoint(self, sess, epoch, summary_str):
        if self.task_index == 0:
            if summary_str:
                self.summary_writer.add_summary(summary_str, global_step=epoch)
            saver = tf.train.Saver(max_to_keep=5)
            saver.save(sess, self.checkpoint_file, epoch)

    def save_model(self, sess):
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, self.model.output_node_names)
        with tf.gfile.FastGFile(self.model_path, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    def execute(self):
        result_file = os.path.join(self.result_dir, "train_result_{}.txt".format(self.task_index))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        summary_op = tf.summary.merge_all()
        with tf.Session(self.server.target, config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            if self.go_on:
                self.restore_checkpoint(sess)
            names, values = zip(*self.model.fetches.items())
            names = list(names)
            values = list(values)
            res, summary_str = None, None
            for epoch in range(1, self.epochs + 1):
                for _ in range(self.steps_per_epoch):
                    x, y = self.generate_rdd_data
                    if len(x) == 0:
                        break
                    if summary_op is not None:
                        *res, summary_str = sess.run(values + [summary_op], self.feed_dict(x=x, y=y))
                    else:
                        res = sess.run(values, self.feed_dict(x=x, y=y))
                    break
                result = dict((k, v) for k, v in zip(names, res) if v is not None)
                result.update(self.common_dict(epoch + self.initial_epoch))
                ModelDir.write_result(result_file, [result], True)
                self.save_checkpoint(sess, epoch + self.initial_epoch, summary_str)

            self.model.write_model(self.model_config_path, False)
            self.save_model(sess)
            self.tf_feed.terminate()


class TFEvaluateWorker(TFWorker):

    def load_model(self, sess):
        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图

    def execute(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(self.server.target, config=config) as sess:
            self.load_model(sess)


class TFPredictWorker(TFWorker):

    def __init__(self, params: (dict, type(None)) = None, *args, **kwargs):
        super(TFPredictWorker, self).__init__(*args, **kwargs)
        self.params = params if params else {}

    def get_results(self, y_pred, y_true=None):
        results = []
        length = len(y_pred)
        if y_true is not None:
            assert length == len(y_true)

        for i in range(length):
            results.append({
                '_task_index': self.task_index,
                'y_pred': y_pred[i],
                'y_true': y_true[i]
            })
        return results

    def load_model(self, sess):
        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            logger.debug(graph_def)
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            self.model = TFCompile.read_model(self.model_config_path)
            self.model.icy = False
            self.model.update_params(self.params)

    def execute(self):
        result_file = os.path.join(self.result_dir, "predict_result_{}.txt".format(self.task_index))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(self.server.target, config=config) as sess:
            self.load_model(sess)
            for _ in range(self.steps_per_epoch):
                x, y = self.generate_rdd_data
                if len(x) == 0:
                    break
                predictions = sess.run(self.model.outputs['y'], self.feed_dict(x=x))
                y_pred = np.argmax(predictions, 1)
                y_true = np.argmax(y, 1) if y is not None else None
                logger.debug(predictions)
                results = self.get_results(y_pred, y_true)
                ModelDir.write_result(result_file, results, True)
            self.tf_feed.terminate()

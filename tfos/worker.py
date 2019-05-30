# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 26/05/2019 19:00
:File    : graph.py
"""

import logging
import os
import time
from collections import namedtuple
from datetime import datetime

import numpy as np
import tensorflow as tf

GRAPH = namedtuple("graph",
                   ['label', 'prediction', 'accuracy', 'loss', 'steps', 'train_op', 'saver', 'summary_op', 'init_op'])


class Worker(object):

    def __init__(self):
        self.args = None
        self.ctx = None
        self.main_func = None
        self.tf_feed = None
        self.mode = 'train'
        self.model_path = None
        self.model_type = 0  # 0 is classifier, 1 is regression
        self.graph = None
        self.step = 0
        self.x_dim = 0
        self.iterator = None
        self.input_mode = 0

    def get_path(self, type):
        TYPES = ['train', 'inference', 'prediction', 'tensorboard', 'checkpoint']

        if type in TYPES:
            path = os.path.join(self.model_path, type)
            # if not os.path.exists(path):
            #     os.makedirs(path)
        else:
            raise ValueError("model save type is error!")
        return path

    def rdd_iterator(self):
        tf_feed = self.ctx.get_data_feed(self.mode == 'train')
        self.tf_feed = tf_feed

        def rdd_generator():
            while not tf_feed.should_stop():
                batch = tf_feed.next_batch(1)
                if len(batch) == 0:
                    return
                row = batch[0]
                data = np.array(row[0]).astype(np.float32)
                label = np.array(row[1]).astype(np.int64)
                yield data, label

        # Dataset for input data
        ds = tf.data.Dataset.from_generator(rdd_generator,
                                            (tf.float32, tf.float32),
                                            (tf.TensorShape([self.x_dim]), tf.TensorShape([1]))
                                            ).batch(self.args.batch_size)
        iterator = ds.make_one_shot_iterator()
        # x, y = iterator.get_next()
        return iterator

    def tf_data(self):
        pass

    def get_iterator(self):
        if self.input_mode == 0:
            return self.rdd_iterator()
        else:
            return self.iterator

    def execute(self):
        worker_num = self.ctx.worker_num
        job_name = self.ctx.job_name
        task_index = self.ctx.task_index

        # Get TF cluster and server instances
        cluster, server = self.ctx.start_cluster_server(1, self.args.rdma)

        if job_name == "ps":
            server.join()
        elif job_name == "worker":
            self.worker(cluster, task_index)
            self.start_session(worker_num, task_index, server.target)

    def worker(self, cluster, task_index):
        # Use between-graph replication distribution, difference graph save same parameters to ps
        with tf.device(
                tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index, cluster=cluster)):
            iterator = self.rdd_iterator()
            x, y = iterator.get_next()
            pred, loss, train_op, steps = self.main_func(x, y)
            accuracy = None
            if self.model_type == 0:
                # Test trained model
                label = tf.argmax(y, 1, name="label")
                prediction = tf.argmax(pred, 1, name="prediction")
                correct_prediction = tf.equal(prediction, label)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
                tf.summary.scalar("accuracy", accuracy)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
            self.graph = GRAPH._make([y, pred, accuracy, loss, steps, train_op, saver, summary_op, init_op])

    def start_session(self, worker_num, task_index, master):
        # Asynchronous training
        g = self.graph
        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        checkpoint_dir = self.ctx.absolute_path(self.get_path('checkpoint'))
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.get_path('tensorboard'), "tensorboard_%d" % worker_num),
            graph=tf.get_default_graph())
        hooks = [tf.train.StopAtStepHook(last_step=self.args.steps)] if self.mode == "train" else []
        with tf.train.MonitoredTrainingSession(master=master,
                                               is_chief=(task_index == 0),
                                               scaffold=tf.train.Scaffold(init_op=g.init_op,
                                                                          summary_op=g.summary_op,
                                                                          saver=g.saver),
                                               checkpoint_dir=checkpoint_dir,
                                               hooks=hooks) as sess:
            # Loop until the session shuts down or feed has no more data
            while not sess.should_stop() and not self.tf_feed.should_stop():
                if self.mode == "train":
                    self.train(sess, task_index, summary_writer)
                else:  # self.mode == "inference":
                    self.inference(sess)

        self.stop_session(sess)

    def train(self, sess, task_index, summary_writer):
        _, summary, self.step = sess.run([self.graph.train_op, self.graph.summary_op, self.graph.steps])
        if (self.step % 2 == 0) and (not sess.should_stop()):
            # print("{} step: {} accuracy: {}".format(datetime.now().isoformat(), step, sess.run(accuracy)))
            if self.model_type == 0:
                logging.error(
                    "{} step: {} accuracy: {}".format(datetime.now().isoformat(), self.step,
                                                      sess.run(self.graph.accuracy)))
            else:
                logging.error(
                    "{} step: {} loss: {}".format(datetime.now().isoformat(), self.step, sess.run(self.graph.loss)))
        if task_index == 0:
            summary_writer.add_summary(summary, self.step)

    def inference(self, sess):
        labels, preds, acc = sess.run([self.graph.label, self.graph.prediction, self.graph.accuracy])
        results = ["{} Label: {}, Prediction: {}".format(datetime.now().isoformat(), l, p) for l, p in
                   zip(labels, preds)]
        self.tf_feed.batch_results(results)

    def stop_session(self, sess):
        print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))
        if sess.should_stop() or self.step >= self.args.steps:
            self.tf_feed.terminate()

            # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
            # wait for all other nodes to complete (via done files)
            done_dir = "{}/{}/done".format(self.ctx.absolute_path(self.args.model_path), self.mode)
            print("Writing done file to: {}".format(done_dir))
            tf.gfile.MakeDirs(done_dir)
            with tf.gfile.GFile("{}/{}".format(done_dir, self.ctx.task_index), 'w') as done_file:
                done_file.write("done")

            for i in range(60):
                if len(tf.gfile.ListDirectory(done_dir)) < len(self.ctx.cluster_spec['worker']):
                    print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
                    time.sleep(1)
                else:
                    print("{} All nodes done".format(datetime.now().isoformat()))
                    break

    def __call__(self, args, ctx):
        self.args = args
        self.ctx = ctx
        self.execute()

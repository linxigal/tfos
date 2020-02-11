#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/20 10:10
:File   :worker.py
:content:
  
"""

import os

import numpy as np
import tensorflow as tf
from facenet.src import facenet
from tensorflowonspark import TFNode

from tfos.base.gfile import ModelDir
from tfos.tf.model import INPUTS, OUTPUTS
from tfos.tf.worker import TFTrainWorker
from tfos.tf.model import TFMode


class FaceNetSoftMaxWorker(TFTrainWorker):
    def __init__(self, n_classes,
                 gpu_memory_fraction,
                 random_rotate,
                 random_crop,
                 random_flip,
                 use_fixed_image_standardization,
                 *args, **kwargs):
        super(FaceNetSoftMaxWorker, self).__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.gpu_memory_fraction = gpu_memory_fraction
        self.random_rotate = random_rotate
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.use_fixed_image_standardization = use_fixed_image_standardization

    def generate_data(self):
        while not self.tf_feed.should_stop():
            batch = self.tf_feed.next_batch(self.batch_size)
            if len(batch) == 0:
                return
            row = batch[0]
            yield row[0], row[1]

    def get_data(self):
        # Dataset for input data
        image_size = self.model.params['image_size']
        n_classes = self.model.params['n_classes']
        ds = tf.data.Dataset.from_generator(self.generate_data, (tf.float32, tf.float32), (
            tf.TensorShape([image_size * image_size]), tf.TensorShape([n_classes]))).batch(
            self.batch_size)
        iterator = ds.make_one_shot_iterator()
        x, y_ = iterator.get_next()
        return x, y_

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:{}".format(self.task_index), cluster=self.cluster)):
            self.model = TFMode().deserialize(self.model_rdd)

    def execute(self):
        result_file = os.path.join(self.result_dir, "train_result_{}.txt".format(self.task_index))
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            steps = 0
            while not sess.should_stop() and not self.tf_feed.should_stop():
                if self.go_on:
                    self.restore_model(sess)

                # Training and validation loop
                print('Running training')
                image_list, label_list = self.get_data()
                # Enqueue one epoch of image paths and labels
                labels_array = np.expand_dims(np.array(image_list), 1)
                image_paths_array = np.expand_dims(np.array(label_list), 1)
                control_value = facenet.RANDOM_ROTATE * self.random_rotate + \
                                facenet.RANDOM_CROP * self.random_crop + \
                                facenet.RANDOM_FLIP * self.random_flip + \
                                facenet.FIXED_STANDARDIZATION * self.use_fixed_image_standardization
                control_array = np.ones_like(labels_array) * control_value
                enqueue_op = tf.get_collection(OUTPUTS)[0]
                feed_dict = dict(zip(tf.get_collection(INPUTS), [image_paths_array, labels_array, control_array]))
                sess.run(enqueue_op, feed_dict)

                self.model.add_params(batch_size=self.batch_size, steps_per_epoch=self.steps_per_epoch,
                                      phase_train=True, n_classes=self.n_classes)
                keys = ["_task_index", "_epoch"]
                for epoch in range(1, self.epochs + 1):
                    for _ in range(self.steps_per_epoch - 1):
                        sess.run(self.model.fetches, feed_dict=self.model.feed_dict)
                    res = sess.run(self.model.fetches + [summary_op], feed_dict=self.model.feed_dict)
                    steps = sess.run(self.global_step)
                    summary_writer.add_summary(res[-1], global_step=steps)
                    results = [dict(zip(keys, res))]
                    ModelDir.write_result(result_file, results, True)
            summary = tf.Summary()
            summary_writer.add_summary(summary, global_step=steps)
            self.tf_feed.terminate()

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

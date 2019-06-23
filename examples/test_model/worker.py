# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from tensorflowonspark import TFNode


class BaseWorker(object):
    def __init__(self, model_config, compile_config, batch_size=1, epochs=1, steps_per_epoch=1, model_dir=None):
        self.compile_config = compile_config
        self.model_config = model_config
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_dir = model_dir
        self.task_index = None
        self.job_name = None
        self.tf_feed = None
        self.cluster = None
        self.server = None
        self.model = None
        self.checkpoint_path = os.path.join(self.model_dir, "checkpoint")
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        # self.checkpoint_file = os.path.join(self.checkpoint_path, 'model_checkpoint.hdf5')
        self.checkpoint_file = os.path.join(self.checkpoint_path, 'model_checkpoint')
        self.tensorboard_path = os.path.join(self.model_dir, "tensorboard")
        self.export_path = os.path.join(self.model_dir, "export")

    def generate_rdd_data(self, tf_feed):
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(self.batch_size)
            inputs = []
            labels = []
            for item in batch:
                inputs.append(item.features)
                labels.append(item.label)
            inputs = np.array(inputs).astype('float32')
            labels = np.array(labels).astype('float32')
            yield (inputs, labels)

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.task_index, cluster=self.cluster)):
            model = Sequential().from_config(self.model_config)

            if os.path.exists(self.checkpoint_file) and self.task_index == 0:
                model.load_weights(self.checkpoint_file)

            model.summary()
            model.compile(**self.compile_config)
        self.model = model
        return model

    def execute_model(self):
        raise NotImplementedError

    def __save_model(self, sess):
        if self.export_dir and self.job_name == 'worker' and self.task_index == 0:
            # save a local Keras model, so we can reload it with an inferencing learning_phase
            if os.path.exists(self.export_dir):
                shutil.rmtree(self.export_dir)

            # save_model(self.model, "tmp_model")
            # # reload the model
            # K.set_learning_phase(False)
            # new_model = load_model("tmp_model")

            # export a saved_model for inferencing
            builder = saved_model_builder.SavedModelBuilder(self.export_dir)
            signature = predict_signature_def(inputs={'images': self.model.input},
                                              outputs={'scores': self.model.output})
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[tag_constants.SERVING],
                                                 signature_def_map={'predict': signature},
                                                 clear_devices=True)
            builder.save()

    def __call__(self, args, ctx):
        self.task_index = ctx.task_index
        self.job_name = ctx.job_name
        self.cluster, self.server = TFNode.start_cluster_server(ctx)
        self.tf_feed = TFNode.DataFeed(ctx.mgr)

        if ctx.job_name == "ps":
            self.server.join()
        elif ctx.job_name == "worker":
            self.build_model()
            self.execute_model()

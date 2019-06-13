#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 16:10
:File       : test_model_train.py
"""

import json
import os

import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import LambdaCallback, TensorBoard, CSVLogger, Callback
from tensorflow.python.keras.models import Sequential, load_model, save_model
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflowonspark import TFCluster, TFNode

from examples.base import *


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class Worker(object):
    def __init__(self, model_config, batch_size, epochs, steps_per_epoch, model_dir, export_dir):
        self.model_config = model_config
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_dir = model_dir
        self.export_dir = export_dir
        self.task_index = None
        self.job_name = None
        self.tf_feed = None
        self.cluster = None
        self.server = None
        self.model = None

    def generate_rdd_data(self, tf_feed):
        while not tf_feed.should_stop():
            batch = tf_feed.next_batch(self.batch_size)
            inputs = []
            labels = []
            for item in batch:
                inputs.append(item[0])
                labels.append(item[1])
            inputs = np.array(inputs).astype('float32')
            labels = np.array(labels).astype('float32')
            yield (inputs, labels)

    def build_model(self):
        if self.task_index is None:
            raise ValueError("task_index cannot None!!!")
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % self.task_index, cluster=self.cluster)):
            model = Sequential().from_config(self.model_config)
            model.summary()
            model.compile(loss='mean_squared_error',
                          optimizer=RMSprop(),
                          metrics=['accuracy'])
        self.model = model
        return model

    def execute_model(self):
        saver = tf.train.Saver()
        with tf.Session(self.server.target) as sess:
            K.set_session(sess)

            def save_checkpoint(epoch, logs=None):
                logging.error(logs)
                if epoch == 1:
                    tf.train.write_graph(sess.graph.as_graph_def(), self.model_dir, 'graph.pbtxt')
                saver.save(sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=epoch * self.steps_per_epoch)

            ckpt_callback = LambdaCallback(on_epoch_end=save_checkpoint)
            csv_logger = CSVLogger(os.path.join(os.path.dirname(self.model_dir), 'train.log'))
            # tb_callback = TensorBoard(log_dir=self.model_dir, histogram_freq=1, write_graph=True, write_images=True)
            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            history = LossHistory()
            callbacks = [ckpt_callback, csv_logger, history] if self.task_index == 0 else None

            # train on data read from a generator which is producing data from a Spark RDD
            self.model.fit_generator(generator=self.generate_rdd_data(self.tf_feed),
                                     steps_per_epoch=self.steps_per_epoch,
                                     epochs=self.epochs,
                                     verbose=1,
                                     # validation_data=(x_test, y_test),
                                     callbacks=callbacks
                                     )
            print(history.losses)

            if self.export_dir and self.job_name == 'worker' and self.task_index == 0:
                # save a local Keras model, so we can reload it with an inferencing learning_phase
                save_model(self.model, "tmp_model")

                # reload the model
                K.set_learning_phase(False)
                new_model = load_model("tmp_model")

                # export a saved_model for inferencing
                builder = saved_model_builder.SavedModelBuilder(self.export_dir)
                signature = predict_signature_def(inputs={'images': new_model.input},
                                                  outputs={'scores': new_model.output})
                builder.add_meta_graph_and_variables(sess=sess,
                                                     tags=[tag_constants.SERVING],
                                                     signature_def_map={'predict': signature},
                                                     clear_devices=True)
                builder.save()

            self.tf_feed.terminate()

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


class TestTrainModel(Base):
    def __init__(self, input_table_name, input_model_config, cluster_size, num_ps, batch_size, epochs,
                 steps_per_epoch, model_dir, export_dir):
        super(TestTrainModel, self).__init__()
        self.p('input_table_name', input_table_name)
        self.p('input_model_config', input_model_config)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        self.p('model_dir', model_dir)
        self.p('export_dir', export_dir)
        self.p('steps_per_epoch', steps_per_epoch)

    def run(self):
        param = self.params

        # param = json.loads('<#zzjzParam#>')
        input_table_name = param.get('input_table_name')
        input_model_config = param.get('input_model_config')
        cluster_size = param.get('cluster_size')
        num_ps = param.get('num_ps')

        batch_size = param.get('batch_size')
        epochs = param.get('epochs')
        steps_per_epoch = param.get('steps_per_epoch')
        model_dir = param.get('model_dir')
        export_dir = param.get('export_dir')

        model_rdd = inputRDD(input_model_config)
        model_config = json.loads(model_rdd.first().model_config)
        print(json.dumps(model_config, indent=4))
        worker = Worker(model_config, batch_size, epochs, steps_per_epoch, model_dir, export_dir)

        input_rdd = inputRDD(input_table_name)

        cluster = TFCluster.run(sc, worker, None, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK)
        cluster.train(input_rdd)


if __name__ == "__main__":
    from examples import ROOT_PATH
    from examples.test_layer.test_dense import TestDense
    from examples.test_data.test_read_csv import TestReadCsv
    from examples.test_data.test_df2data import TestDF2Inputs

    filepath = os.path.join(ROOT_PATH, 'output_data', 'data', 'regression_data.csv')
    TestReadCsv(filepath).run()
    TestDF2Inputs('<#zzjzRddName#>', '5').run()
    TestDense("<#zzjzRddName#>_dense", 1, input_dim=5).run()
    model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")
    export_dir = os.path.join(ROOT_PATH, 'output_data', "export_dir")
    TestTrainModel('<#zzjzRddName#>', '<#zzjzRddName#>_dense', 2, 1, 2, 2, 3, model_dir, export_dir).run()

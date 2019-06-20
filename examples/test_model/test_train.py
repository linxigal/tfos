#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 16:10
:File       : test_model_train.py
"""

import json
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, TensorBoard, LambdaCallback
from tensorflow.python.keras.models import Sequential
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
    def __init__(self, model_config, compile_config, batch_size, epochs, steps_per_epoch, model_dir):
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
            model.summary()
            model.compile(**self.compile_config)
        self.model = model
        return model

    def execute_model(self):
        saver = tf.train.Saver()
        with tf.Session(self.server.target) as sess:
            # K.set_session(sess)

            def save_checkpoint(epoch, logs=None):
                saver.save(sess, os.path.join(self.checkpoint_path, 'model.ckpt'),
                           global_step=(epoch + 1) * self.steps_per_epoch)

            ckpt_callback = LambdaCallback(on_epoch_end=save_checkpoint)

            # tb_callback = TensorBoard(log_dir=self.tensorboard_path, histogram_freq=1, write_graph=True,
            #                           write_images=True)
            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            callbacks = [ckpt_callback] if self.task_index == 0 else None

            # train on data read from a generator which is producing data from a Spark RDD
            self.model.fit_generator(generator=self.generate_rdd_data(self.tf_feed),
                                     steps_per_epoch=self.steps_per_epoch,
                                     epochs=self.epochs,
                                     verbose=2,
                                     # validation_data=(x_test, y_test),
                                     callbacks=callbacks
                                     )
            # self.__save_model(sess)
            self.tf_feed.terminate()

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


class TestTrainModel(Base):
    def __init__(self, input_rdd_name, input_config, cluster_size, num_ps, batch_size,
                 epochs,
                 steps_per_epoch,
                 model_dir):
        super(TestTrainModel, self).__init__()
        self.p('input_rdd_name', input_rdd_name)
        self.p('input_config', input_config)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        self.p('steps_per_epoch', steps_per_epoch)
        self.p('model_dir', model_dir)

    def run(self):
        param = self.params

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        input_config = param.get('input_config')
        cluster_size = param.get('cluster_size')
        num_ps = param.get('num_ps')

        batch_size = param.get('batch_size')
        epochs = param.get('epochs')
        steps_per_epoch = param.get('steps_per_epoch')
        model_dir = param.get('model_dir')

        # load data
        assert input_rdd_name, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        # print(input_rdd.rdd.take(1))
        # load config
        assert input_config, "parameter input_model_config cannot empty!"
        model_config_rdd = inputRDD(input_config)
        assert model_config_rdd, "cannot get model config rdd from previous model layer!"
        columns = model_config_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        model_config = json.loads(model_config_rdd.first().model_config)
        compile_config = json.loads(model_config_rdd.first().compile_config)
        # print(json.dumps(model_config, indent=4))
        # print(json.dumps(compile_config, indent=4))
        worker = Worker(model_config, compile_config, batch_size, epochs, steps_per_epoch, model_dir)

        cluster = TFCluster.run(sc, worker, None, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK)
        cluster.train(input_rdd.rdd)


if __name__ == "__main__":
    from examples import ROOT_PATH
    from examples.test_layer.test_dense import TestDense
    from examples.test_data.test_read_csv import TestReadCsv
    from examples.test_data.test_df2data import TestDF2Inputs
    from examples.test_optimizer.test_optimizer import TestOptimizer

    # load data
    filepath = os.path.join(ROOT_PATH, 'output_data', 'data', 'regression_data.csv')
    TestReadCsv(filepath).run()
    TestDF2Inputs('<#zzjzRddName#>', '5').run()

    # build model
    TestDense("<#zzjzRddName#>_model_config", 1, input_dim=5).run()

    # compile model
    output_compile_name = "<#zzjzRddName#>_model_config"
    TestOptimizer(output_compile_name, 'mse', 'rmsprop',
                  ['accuracy']
                  ).run()

    # train model
    model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")
    TestTrainModel('<#zzjzRddName#>', '<#zzjzRddName#>_model_config',
                   cluster_size=2,
                   num_ps=1,
                   batch_size=1,
                   epochs=5,
                   steps_per_epoch=5,
                   model_dir=model_dir).run()

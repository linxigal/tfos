#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 16:10
:File       : test_model_train.py
"""

import json
import os

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint

from examples.base import *
from examples.test_model.worker import BaseWorker
from tensorflowonspark import TFCluster, TFNode


class Worker(BaseWorker):

    def execute_model(self):
        with tf.Session(self.server.target) as sess:
            sess.run(tf.global_variables_initializer())

            tb_callback = TensorBoard(log_dir=self.tensorboard_path, write_grads=True, write_images=True)
            ckpt_callback = ModelCheckpoint(self.checkpoint_file, save_weights_only=True)

            # add callbacks to save model checkpoint and tensorboard events (on worker:0 only)
            callbacks = [tb_callback, ckpt_callback] if self.task_index == 0 else None

            # train on data read from a generator which is producing data from a Spark RDD
            self.model.fit_generator(generator=self.generate_rdd_data(),
                                     steps_per_epoch=self.steps_per_epoch,
                                     epochs=self.epochs,
                                     callbacks=callbacks
                                     )
            # self.__save_model(sess)
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
    def __init__(self, input_rdd_name, input_config, cluster_size, num_ps, batch_size,
                 epochs,
                 model_dir):
        super(TestTrainModel, self).__init__()
        self.p('input_rdd_name', input_rdd_name)
        self.p('input_config', input_config)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        # self.p('steps_per_epoch', steps_per_epoch)
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
        # steps_per_epoch = param.get('steps_per_epoch')
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
        n_samples = input_rdd.count()
        # steps_per_epoch = n_samples // batch_size
        steps_per_epoch = 5
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
                   # steps_per_epoch=5,
                   model_dir=model_dir).run()

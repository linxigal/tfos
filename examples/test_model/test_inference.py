#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/18 15:36
:File   : test_inference.py
"""
import json
import tensorflow as tf
from tensorflowonspark import TFCluster
from examples.base import *
from examples.test_model.worker import BaseWorker


class Worker(BaseWorker):
    def execute_model(self):
        with tf.Session(self.server.target) as sess:
            sess.run(tf.global_variables_initializer())
            self.model.evaluate_generator(generator=self.generate_rdd_data(self.tf_feed),)
            self.tf_feed.terminate()


class TestInferenceModel(Base):
    def __init__(self, input_rdd_name, input_config, cluster_size, num_ps, batch_size,
                 epochs,
                 # steps_per_epoch,
                 model_dir):
        super(TestInferenceModel, self).__init__()
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
        steps_per_epoch = n_samples // batch_size
        worker = Worker(model_config, compile_config, batch_size, epochs, steps_per_epoch, model_dir)
        cluster = TFCluster.run(sc, worker, None, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK)
        ouput_rdd = cluster.inference(input_rdd.rdd)
        ouput_rdd.show()
        outputRDD('<#zzjzRddName#>', ouput_rdd)

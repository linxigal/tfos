#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/18 15:36
:File   : test_inference.py
"""
import json

import tensorflow as tf
from pyspark.sql import Row
from tensorflowonspark import TFCluster

from examples.base import *
from examples.test_model.worker import BaseWorker


class Worker(BaseWorker):
    def execute_model(self):
        with tf.Session(self.server.target) as sess:
            for i in range(self.steps_per_epoch):
                x, y = next(self.generate_rdd_data())
                result = self.model.evaluate(x, y, self.batch_size)
                # numpy.float32 cannot convert to DataFrame
                self.tf_feed.batch_results([Row(loss=float(result[0]), acc=float(result[1]))])
                # self.tf_feed.batch_results([result])
            self.tf_feed.terminate()


class TestInferenceModel(Base):
    def __init__(self, input_rdd_name, input_config, cluster_size, num_ps, model_dir):
        super(TestInferenceModel, self).__init__()
        self.p('input_rdd_name', input_rdd_name)
        self.p('input_config', input_config)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('model_dir', model_dir)

    def run(self):
        param = self.params

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        input_config = param.get('input_config')
        cluster_size = param.get('cluster_size')
        num_ps = param.get('num_ps')

        model_dir = param.get('model_dir')

        # load data
        assert input_rdd_name, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        assert input_config, "parameter input_model_config cannot empty!"
        model_config_rdd = inputRDD(input_config)
        assert model_config_rdd, "cannot get model config rdd from previous model layer!"
        columns = model_config_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        model_config = json.loads(model_config_rdd.first().model_config)
        compile_config = json.loads(model_config_rdd.first().compile_config)
        n_samples = input_rdd.count()
        steps_per_epoch = n_samples
        # steps_per_epoch = 5
        worker = Worker(model_config, compile_config, steps_per_epoch=steps_per_epoch, model_dir=model_dir)
        cluster = TFCluster.run(sc, worker, None, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK)
        output_rdd = cluster.inference(input_rdd.rdd)
        output_rdd = output_rdd.toDF()
        output_rdd.show()
        outputRDD('<#zzjzRddName#>', output_rdd)

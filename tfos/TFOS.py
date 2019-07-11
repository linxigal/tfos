#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/5/24 14:01
:File       : TFOS.py
"""

import json
from tensorflowonspark import TFCluster

from .worker import Worker


class TFOS(object):

    def __init__(self, sc):
        self.sc = sc

    def train(self, data_rdd, model_rdd, cluster_size, num_ps, batch_size, epochs, model_dir):
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        model_config = json.loads(model_rdd.first().model_config)
        compile_config = json.loads(model_rdd.first().compile_config)
        n_samples = data_rdd.count()
        # steps_per_epoch = n_samples // batch_size
        steps_per_epoch = 1
        worker = Worker(model_config, compile_config, batch_size, epochs, steps_per_epoch, model_dir)
        cluster = TFCluster.run(self.sc, worker, None, cluster_size, num_ps, input_mode=TFCluster.InputMode.SPARK)
        cluster.train(data_rdd.rdd)

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 16:10
:File       : test_model_train.py
"""

import os
from examples.base import *


class TestTrainModel(Base):
    def __init__(self, input_rdd_name, input_config, cluster_size, num_ps, batch_size, epochs, model_dir):
        super(TestTrainModel, self).__init__()
        self.p('input_rdd_name', input_rdd_name)
        self.p('input_config', input_config)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('batch_size', batch_size)
        self.p('epochs', epochs)
        # self.p('steps_per_epoch', steps_per_epoch)
        self.p('model_dir', [{"path": model_dir}])

    def run(self):
        param = self.params

        from tfos import TFOS

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        input_config = param.get('input_config')
        cluster_size = int(param.get('cluster_size'))
        num_ps = int(param.get('num_ps'))
        batch_size = int(param.get('batch_size'))
        epochs = int(param.get('epochs'))
        model_dir = param.get('model_dir')[0]['path']

        # load data
        assert input_rdd_name, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        # load model
        assert input_config, "parameter input_model_config cannot empty!"
        model_rdd = inputRDD(input_config)
        assert model_rdd, "cannot get model config rdd from previous model layer!"
        TFOS(sc).train(input_rdd, model_rdd, cluster_size, num_ps, batch_size, epochs, model_dir)


if __name__ == "__main__":
    from examples import ROOT_PATH
    from examples.test_layer import TestDense
    from examples.test_data import TestReadCsv, TestDF2Inputs
    from examples.test_optimizer import TestOptimizer

    # load data
    filepath = os.path.join(ROOT_PATH, 'output_data', 'data', 'regression_data.csv')
    TestReadCsv(filepath).run()
    TestDF2Inputs('<#zzjzRddName#>_data', '5').run()

    # build model
    TestDense(lrn(), 1, input_dim=5).run()
    # compile model
    TestOptimizer(lrn(), 'mse', 'rmsprop', ['accuracy']).run()
    # train model
    model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")
    TestTrainModel('<#zzjzRddName#>_data', lrn(),
                   cluster_size=2,
                   num_ps=1,
                   batch_size=1,
                   epochs=5,
                   model_dir=model_dir).run()

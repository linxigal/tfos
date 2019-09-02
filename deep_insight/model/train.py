#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 16:10
:File       : train.py

:content

    spark-submit    --master ${MASTER} \
                        --num-executors 2 \
                        --executor-cores 1\
                        --executor-memory 16G \
                        deep_insight/model/train.py
"""

import os
import unittest

from deep_insight import ROOT_PATH
from deep_insight.base import *
from deep_insight.data.read_mnist import ReadMnist
from deep_insight.layers import Dropout, Dense
from deep_insight.optimizers import Optimizer


class TrainModel(Base):
    """模型训练
    神经网络模型训练算子

    参数：
        input_rdd_name: 输入数据
            输入的RDD格式数据
        input_config: 输入模型
            输入的RDD模型配置数据，包括模型的图结构以及模型的编译优化参数
        cluster_size: 集群数量
            tensorflow集群数量，包括参数服务器ps和计算服务器worker
        num_ps: 参数服务器数量
        batch_size： 批处理大小
        epochs：数据集迭代次数
            对传入的数据集，训练模型时需要迭代的次数
        model_dir: 模型保存路径
            保存路径下会自动生成tensorboard目录，checkpoint目录以及save_model目录
    """

    def __init__(self, input_rdd_name, input_config, cluster_size, num_ps, batch_size, epochs, model_dir):
        super(TrainModel, self).__init__()
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
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        batch_size = param.get('batch_size', 32)
        epochs = param.get('epochs', 1)
        model_dir = param.get('model_dir')[0]['path']

        # param check
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)
        batch_size = int(batch_size)
        epochs = int(epochs)

        # load data
        assert input_rdd_name, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        # load model
        assert input_config, "parameter input_model_config cannot empty!"
        model_rdd = inputRDD(input_config)
        assert model_rdd, "cannot get model config rdd from previous model layer!"
        TFOS(sc, cluster_size, num_ps).train(input_rdd, model_rdd, batch_size, epochs, model_dir)
        outputRDD('<#zzjzRddName#>_model', model_rdd)


class TestTrainModel(unittest.TestCase):
    # @unittest.skip("")
    def test_train_model(self):
        # load data
        output_data_name = "<#zzjzRddName#>_data"
        # build model
        Dense(lrn(), 512, activation='relu', input_dim=784).run()
        Dropout(lrn(), 0.2).run()
        Dense(lrn(), 512, activation='relu').run()
        Dropout(lrn(), 0.2).run()
        Dense(lrn(), 10, activation='softmax').run()

        # compile model
        Optimizer(lrn(), 'categorical_crossentropy', 'rmsprop', ['accuracy']).run()

        # show network struct
        SummaryLayer(lrn()).run()

        input_path = os.path.join(ROOT_PATH, 'data/mnist/tfr/test')
        model_dir = os.path.join(ROOT_PATH, 'data/model_dir')

        ReadMnist(input_path, 'tfr').run()
        TrainModel(output_data_name, lrn(),
                   cluster_size=3,
                   num_ps=1,
                   batch_size=32,
                   epochs=1,
                   model_dir=model_dir).run()


if __name__ == '__main__':
    unittest.main()

# if __name__ == "__main__":
#     from deep_insight import ROOT_PATH
#     from deep_insight.layers import Dense
#     from deep_insight.data import ReadCsv, DF2Inputs
#     from deep_insight.optimizers import Optimizer
#
#     # load data
#     filepath = os.path.join(ROOT_PATH, 'output_data', 'data', 'regression_data.csv')
#     ReadCsv(filepath).run()
#     DF2Inputs('<#zzjzRddName#>_data', '5').run()
#
#     # build model
#     Dense(lrn(), 1, input_dim=5).run()
#     # compile model
#     Optimizer(lrn(), 'mse', 'rmsprop', ['accuracy']).run()
#     # train model
#     model_dir = os.path.join(ROOT_PATH, 'output_data', "model_dir")
#     TrainModel('<#zzjzRddName#>_data', lrn(),
#                cluster_size=2,
#                num_ps=1,
#                batch_size=1,
#                epochs=5,
#                model_dir=model_dir).run()

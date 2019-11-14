#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/11/13 8:43
:File   :RecurrentModel.py
:content:
  
"""

import unittest

from deep_insight import *
from deep_insight.base import *
from deep_insight.data.mnist import Mnist


class RecurrentPredictModel(Base):
    """模型循环预测

    参数：
        cluster_size: 集群数量
            tensorflow集群数量，包括参数服务ps和计算服务worker
        num_ps: 参数服务器数量
        units: 输入单元
            输入数据的维度
        steps: 预测步数
            往后预测的步骤数
        model_dir: 模型目录
            模型的保存路径
    """

    def __init__(self, cluster_size, num_ps, units, steps, model_dir, **kwargs):
        super(RecurrentPredictModel, self).__init__(**kwargs)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('units', units)
        self.p('steps', steps)
        self.p('model_dir', [{"path": model_dir}])

    def run(self):
        param = self.params

        from tfos import TFOS

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        cluster_size = param.get('cluster_size', 3)
        num_ps = param.get('num_ps', 1)
        units = param.get('units', 0)
        steps = param.get('steps', 0)
        model_dir = param.get('model_dir')[0]['path']

        # param check
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)
        steps = int(steps)

        # load data
        assert input_rdd_name is not None, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"

        output_df = TFOS(sc, sqlc, cluster_size, num_ps).recurrent_predict(input_rdd, units, steps, model_dir)
        if output_df:
            output_df.show()
            outputRDD('<#zzjzRddName#>_recurrent_predict_result', output_df)


class TestRecurrentModel(unittest.TestCase):

    def setUp(self) -> None:
        self.is_local = True
        self.mnist_dir = os.path.join(self.path, 'data/data/mnist')
        self.model_dir = os.path.join(self.path, 'data/model/mnist_mlp')

    def tearDown(self) -> None:
        reset()

    @property
    def path(self):
        return ROOT_PATH if self.is_local else HDFS

    # @unittest.skip("")
    def test_predict_model(self):
        Mnist(self.mnist_dir, mode='test').b(DATA_BRANCH).run()
        RecurrentPredictModel(input_prev_layers=MODEL_BRANCH,
                              input_rdd_name=DATA_BRANCH,
                              cluster_size=3,
                              num_ps=1,
                              units=784,
                              steps=10,
                              model_dir=self.model_dir).run()


if __name__ == '__main__':
    unittest.main()

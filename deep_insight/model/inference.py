#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/18 15:36
:File   : inference.py
"""

import unittest
from deep_insight.base import *


class InferenceModel(Base):
    """模型校验
    神经网络模型校验算子

    参数：
        input_rdd_name: 输入数据
            输入rdd校验数据
        input_config: 输入模型
            输入的RDD模型配置数据，包括模型的图结构以及模型的编译优化参数
        cluster_size: 集群数量
            tensorflow集群数量，包括参数服务器ps和计算服务器worker
        num_ps: 参数服务器数量
        model_dir: 模型保存路径
            保存路径下会自动生成tensorboard目录，checkpoint目录以及save_model目录
    """

    def __init__(self, input_rdd_name, input_config, cluster_size, num_ps, model_dir):
        super(InferenceModel, self).__init__()
        self.p('input_rdd_name', input_rdd_name)
        self.p('input_config', input_config)
        self.p('cluster_size', cluster_size)
        self.p('num_ps', num_ps)
        self.p('model_dir', [{"path": model_dir}])

    def run(self):
        param = self.params

        from tfos import TFOS

        # param = json.loads('<#zzjzParam#>')
        input_rdd_name = param.get('input_rdd_name')
        input_config = param.get('input_config')
        cluster_size = param.get('cluster_size')
        num_ps = param.get('num_ps')
        model_dir = param.get('model_dir')[0]['path']

        # param check
        cluster_size = int(cluster_size)
        num_ps = int(num_ps)

        assert input_rdd_name, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        assert input_config, "parameter input_model_config cannot empty!"
        model_rdd = inputRDD(input_config)
        assert model_rdd, "cannot get model config rdd from previous model layer!"
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        output_df = TFOS(sc, cluster_size, num_ps).inference(input_rdd, model_rdd, model_dir)
        output_df.show()
        outputRDD('<#zzjzRddName#>', output_df)


class TestInferenceModel(unittest.TestCase):

    @unittest.skip("")
    def test_inference_model(self):
        pass

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/6/18 15:36
:File   : inference.py
"""
from deep_insight.base import *


class InferenceModel(Base):
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
        cluster_size = int(param.get('cluster_size'))
        num_ps = int(param.get('num_ps'))
        model_dir = param.get('model_dir')[0]['path']

        assert input_rdd_name, "parameter input_rdd_name cannot empty!"
        input_rdd = inputRDD(input_rdd_name)
        assert input_rdd, "cannot get rdd data from previous input layer!"
        assert input_config, "parameter input_model_config cannot empty!"
        model_rdd = inputRDD(input_config)
        assert model_rdd, "cannot get model config rdd from previous model layer!"
        columns = model_rdd.columns
        assert "model_config" in columns, "not exists model layer config!"
        assert "compile_config" in columns, "not exists model compile config!"
        output_df = TFOS(sc).inference(input_rdd, model_rdd, cluster_size, num_ps, model_dir)
        output_df.show()
        outputRDD('<#zzjzRddName#>', output_df)

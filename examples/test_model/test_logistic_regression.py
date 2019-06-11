#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/11 13:07
:File       : test_logistic_regression.py
"""


from examples.base import *


class TestLogisticRegression(Base):
    def __init__(self, inputMutiLayerConfig, output_shape=512, activation='relu', input_shape=(784,)):
        super(TestLogisticRegression, self).__init__()
        self.p('inputMutiLayerConfig', inputMutiLayerConfig)
        self.p('output_shape', output_shape)
        self.p('activation', activation)
        self.p('input_shape', input_shape)

    def run(self):
        param = self.params

        import json
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense
        from pyspark.sql import Row

        # param = json.loads('<#zzjzParam#>')
        inputMutiLayerConfig = param.get("inputMutiLayerConfig")
        output_dims = param.get('output_shape')
        activation = param.get('activation')
        input_shape = param.get('input_shape')
        model_config = inputRDD(inputMutiLayerConfig)

        if model_config:
            model_config = json.loads(model_config.first()._1)
        else:
            model_config = {}
            if not input_shape:
                raise ValueError("当前节点处理网络第一层，输入shape不能为空！！！")

        model = Sequential.from_config(model_config)

        if input_shape:
            model.add(Dense(output_dims, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(output_dims, activation=activation))
        # model.add(Dropout(0.2))

        outputdf = sqlc.createDataFrame([Row(json.dumps(model.get_config()))])
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(inputMutiLayerConfig, outputdf)


if __name__ == "__main__":
    TestLogisticRegression('<#zzjzRddName#>_dense_1').run()
    print_pretty()

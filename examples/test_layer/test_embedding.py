#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 15:28
:File       : test_embedding.py
"""

from examples.base import *


class TestEmbedding(Base):
    def __init__(self, inputMutiLayerConfig, input_dim, output_dim):
        super(TestEmbedding, self).__init__()
        self.p('inputMutiLayerConfig', inputMutiLayerConfig)
        self.p('input_dim', input_dim)
        self.p('output_dim', output_dim)

    def run(self):
        param = self.params

        """This layer can only be used as the first layer in a model.
        """
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Embedding

        # param = json.loads('<#zzjzParam#>')
        inputMutiLayerConfig = param.get("inputMutiLayerConfig")
        input_dim = param.get("input_dim")
        output_dim = param.get("output_dim")

        model_rdd = inputRDD(inputMutiLayerConfig)
        model_config = get_model_config(model_rdd, input_dim=input_dim)
        model = Sequential.from_config(model_config)

        model.add(Embedding(input_dim, output_dim))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(inputMutiLayerConfig, outputdf)


if __name__ == "__main__":
    TestEmbedding('<#zzjzRddName#>', 1024, 256).run()
    TestEmbedding('<#zzjzRddName#>', 2048, 512).run()
    print_pretty('<#zzjzRddName#>')

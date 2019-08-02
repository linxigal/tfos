#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author     :weijinlong
:Time: 2019/6/13 15:28
:File       : embedding.py
"""

from deep_insight.base import *


class Embedding(Base):
    def __init__(self, input_model_config_name, input_dim, output_dim):
        super(Embedding, self).__init__()
        self.p('input_model_config_name', input_model_config_name)
        self.p('input_dim', input_dim)
        self.p('output_dim', output_dim)

    def run(self):
        param = self.params

        """This layer can only be used as the first layer in a model.
        """
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Embedding

        # param = json.loads('<#zzjzParam#>')
        input_model_config_name = param.get("input_model_config_name")
        input_dim = param.get("input_dim")
        output_dim = param.get("output_dim")

        model_rdd = inputRDD(input_model_config_name)
        model_config = get_model_config(model_rdd, input_dim=input_dim)
        model = Sequential.from_config(model_config)

        model.add(Embedding(input_dim, output_dim))

        outputdf = model2df(model)
        # outputRDD('<#zzjzRddName#>_dense', outputdf)
        outputRDD(input_model_config_name, outputdf)


if __name__ == "__main__":
    Embedding('<#zzjzRddName#>', 1024, 256).run()
    Embedding('<#zzjzRddName#>', 2048, 512).run()
    print_pretty('<#zzjzRddName#>')

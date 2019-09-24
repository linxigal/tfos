#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/23 17:03
:File   :test_model.py
:content:
  
"""

from tensorflow.python.keras.layers import InputLayer, Dense, Add, Input
from tensorflow.python.keras.models import Model, Sequential


def network():
    input1 = Input((784,))
    input2 = Input((256,))

    dense1 = Dense(666)(input1)
    dense2 = Dense(666)(input2)

    add = Add()([dense1, dense2])

    model = Model(inputs=[input1, input2], outputs=add)
    # print(model.outputs)
    print(model.inputs)


if __name__ == '__main__':
    network()

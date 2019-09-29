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
from tensorflow.python.keras.optimizers import SGD, serialize, deserialize


def network():
    input1 = Input((784,))
    input2 = Input((256,))

    dense1 = Dense(666)(input1)
    dense2 = Dense(666)(input2)

    add = Add()([dense1, dense2])

    model = Model(inputs=[input1, input2], outputs=add)
    # print(model.outputs)
    print(model.inputs)


def optimizer():
    print(serialize(SGD()))
    print(deserialize(serialize(SGD())))


def classes():
    class A(object):
        def __init__(self, a):
            self.a = a
            print(a)

    class B(object):
        def __init__(self, b):
            self.b = b
            print(b)

    class C(A, B):
        def __init__(self, **kwargs):
            super(C, self).__init__(**kwargs)

    C(a=5, b=6)


if __name__ == '__main__':
    # network()
    # optimizer()
    classes()

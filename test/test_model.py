#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/23 17:03
:File   :test_model.py
:content:
  
"""
import json
from tensorflow.python.keras.layers import InputLayer, Dense, Add, Input, Dropout
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import SGD, serialize, deserialize


def network():
    input1 = Input((784,))
    input2 = Input((256,))

    dense1 = Dense(666)(input1)
    dense2 = Dense(666)(input2)

    drop1 = Dropout(0.5)(dense1)
    dense3 = Dense(666)(drop1)

    add = Add()([dense3, dense2])

    # model = Model(inputs=input1, outputs=dense3)
    model = Model(inputs=[input1, input2], outputs=add)
    # model1 = Model(inputs=input1, outputs=input1)
    # model2 = Model(inputs=input2, outputs=dense2)
    # print(model.outputs)
    # print(model.inputs)
    print("{:*^100}".format('network'))
    config = model.get_config()
    print(len(config))
    print(json.dumps(config, indent=4))


def sequence():
    model = Sequential()
    model.add(Dense(666, input_shape=(784, )))
    model.add(Dense(256))
    print("{:*^100}".format('sequence'))
    config = model.get_config()
    print(len(config))
    print(json.dumps(config, indent=4))


def optimizer():
    # print(serialize(SGD()))
    # print(deserialize(serialize(SGD())))
    sgd = SGD()
    print("{:*^100}".format('optimizer'))
    print(json.dumps(serialize(SGD()), indent=4))


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
    network()
    sequence()
    # optimizer()
    # classes()

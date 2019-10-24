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
    d = Dense(123)
    d._init_set_name('test')
    print(d.get_config())

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
    # print(json.dumps(config, indent=4))
    # model = Sequential.from_config(config)
    # model.summary()
    # print(model.layers)


def sequence():
    model = Sequential()
    model.add(Dense(666, input_shape=(784, )))
    model.add(Dense(256))
    print("{:*^100}".format('sequence'))
    config = model.get_config()
    print(len(config))
    # print(json.dumps(config, indent=4))
    # model = Model.from_config(config)
    # model.summary()


def optimizer():
    # print(serialize(SGD()))
    # print(deserialize(serialize(SGD())))
    sgd = SGD()
    print("{:*^100}".format('optimizer'))
    print(json.dumps(serialize(SGD()), indent=4))


def classes():
    class A(object):
        def __init__(self, a, **kwargs):
            self.a = a
            print(a)

    class B(object):
        def __init__(self, b, **kwargs):
            self.b = b
            print(b)

    class C(A, B):
        def __init__(self, **kwargs):
            super(C, self).__init__(**kwargs)

    C(a=5, b=6)


def merge():
    # model1
    input1 = Input((784,))
    dense = Dense(256)(input1)
    drop = Dropout(0.5)(dense)
    dense1 = Dense(64)(drop)
    model1 = Model(inputs=input1, outputs=dense1)
    # print(json.dumps(model1.get_config(), indent=4))

    # model2
    model2 = Sequential()
    # model2.add(InputLayer((784,)))
    model2.add(Dense(128, input_shape=(64,)))
    # model2.add(Dense(128))
    print(model2.layers)
    print(model2.get_layer(index=0))
    # model2.inputs = model1.output
    # model2.build((None, 784))
    # print(model2.call(Input((784, ))))
    # print(model2.inputs)
    # print(model2.outputs)
    print(json.dumps(model2.get_config(), indent=4))
    # print(model1.outputs)
    # print(model2.outputs)
    # after = Model(inputs=model1.outputs, outputs=model2.outputs)
    # model = Model(inputs=input1, outputs=after.outputs)
    # model.summary()

    # model3
    # model3 = Model(inputs=model1.inputs, outputs=model2.outputs)
    # print(json.dumps(model3.get_config(), indent=4))


if __name__ == '__main__':
    # network()
    # sequence()
    # optimizer()
    # classes()
    merge()

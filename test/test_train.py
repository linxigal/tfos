#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/18 8:59
:File   :test_train.py
:content:
  
"""

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.data.ops.dataset_ops import DatasetV2, DatasetV1Adapter

PATH = "E:\\data\\mnist\\mnist.npz"
(x_train, y_train), (x_test, y_test) = mnist.load_data(PATH)
num = 5
x_train = x_train[:num * 32]
y_train = y_train[:num * 32]


def convert_one_hot(y, num_class):
    if len(y.shape) == 0:
        label = np.zeros(num_class)
        label[y] = 1
    else:
        num = len(y)
        label = np.zeros((num, num_class))
        index = np.array(range(num)) * num_class + y
        label.flat[index] = 1
    return label


def generate_rdd_data():
    count = 0
    batch_size = 32
    while True:
        count += 1
        start, end = (count - 1) * batch_size, count * batch_size
        x = x_train[start:end]
        y = y_train[start:end]
        print("{:*^100}, {}".format(count, x.shape))
        if x.tolist():
            x = x.reshape(32, 784)
            y = convert_one_hot(y, 10)
        else:
            x = np.array(x.tolist())
            y = np.array(y.tolist())
        yield x, y


def build_model():
    input = Input((784,))
    dense = Dense(10, activation='relu')(input)
    model = Model(inputs=input, outputs=dense)
    model.compile('rmsprop', 'categorical_crossentropy', ['accuracy'])
    return model


if __name__ == '__main__':
    generate = generate_rdd_data()
    dataset = tf.data.Dataset.from_generator(generate_rdd_data,
                                             output_types=(tf.float32, tf.float32),
                                             output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]))
                                             )
    model = build_model()
    # model.fit_generator(dataset.make_one_shot_iterator(), steps_per_epoch=5)
    model.fit_generator(generate, steps_per_epoch=5)
    # print(isinstance(dataset, DatasetV2))

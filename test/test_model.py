#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/23 17:03
:File   :test_model.py
:content:
  
"""
import tensorflow as tf
from deep_insight import *
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputLayer, Dense, Add, Input, Dropout
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.datasets.mnist import load_data
from tensorflow.python.keras.utils import to_categorical


def build_model():
    input_data = Input((784,))
    dense1 = Dense(512, activation='relu')(input_data)
    dropout1 = Dropout(0.2)(dense1)
    dense2 = Dense(512, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    dense3 = Dense(10)(dropout2)
    model = Model(inputs=input_data, outputs=dense3)
    model.compile('adadelta', 'categorical_crossentropy', ['accuracy'])
    # model.compile('rmsprop', 'categorical_crossentropy', ['accuracy'])
    # model.summary()
    return model


def train_model(model):
    path = os.path.join(ROOT_PATH, 'data/data/mnist/npz/mnist.npz')
    save_path = os.path.join(ROOT_PATH, 'data/model/mnist_mlp/save_model/model.h5')
    (x_train, y_train), (x_test, y_test) = load_data(path)

    x_train = x_train.reshape(-1, 784) / 255.0
    y_train = to_categorical(y_train, 10)

    with tf.Session() as sess:
        K.set_session(sess)
        for i in range(1):
            his = model.fit(x_train, y_train, 32)
            print(his.history)
        model.save(save_path)


def evaluate_model():
    path = os.path.join(ROOT_PATH, 'data/data/mnist/npz/mnist.npz')
    # save_path = os.path.join(ROOT_PATH, 'data/model/MLP/mlp.h5')
    save_path = os.path.join(ROOT_PATH, 'data/model/mnist_mlp/save_model/model.h5')
    (x_train, y_train), (x_test, y_test) = load_data(path)

    x_test = x_train.reshape(-1, 784) / 255
    y_test = to_categorical(y_train, 10)

    with tf.Session() as sess:
        K.set_session(sess)
        model = load_model(save_path)
        his = model.evaluate(x_test, y_test)
        print(his)


if __name__ == '__main__':
    model = build_model()
    train_model(model)
    # evaluate_model()

#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
:Author :huangdehua
:Time:  :2019/12/06 17:22
:File   : inception_resnet_v2.py
"""

import os
import json
from pyspark.sql import Row

from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation, Reshape
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras import backend
from tensorflow.python.keras.layers.convolutional import MaxPooling2D, Conv2D, AveragePooling2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import np_utils
import numpy as np
# from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import tensorflow as t

from tfos.base import CustomEncoder
from tfos.base import *


# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, bytes):
#             return str(obj, encoding='utf-8');
#         if isinstance(obj, (bytes, bytearray)):
#             return obj.decode("ASCII")  # <- or any other encoding of your choice
#             # Let the base class default method raise the TypeError
#         return json.JSONEncoder.default(self, obj)


class InceptionResnetV2(BaseLayer):

    @ext_exception('InceptionResnetV2 model')
    def add(self, input_shape, reshape, out_dense):
        model = create_inception_resnet_v2(input_shape, reshape, out_dense)
        return self.model2df(model)


def inception_resnet_stem(input):
    channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    c = Conv2D(32, (3, 3), activation='relu', strides=(2, 2))(input)
    c = Conv2D(32, (3, 3), activation='relu', )(c)
    c = Conv2D(64, (3, 3), activation='relu', padding='same')(c)

    # print("c shape : ",c.shape)
    c1 = MaxPooling2D((3, 3), strides=(2, 2))(c)
    # print("c1 shape : ",c1.shape)
    c2 = Conv2D(96, (3, 3), activation='relu', strides=(2, 2))(c)
    # print("c2 shape :",c2.shape)

    m = merge.concatenate([c1, c2], axis=channel_axis)
    # print("m shape :",m.shape)
    # print("#########################")

    c1 = Conv2D(64, (1, 1), activation='relu', padding='same')(m)
    c1 = Conv2D(96, (3, 3), activation='relu', )(c1)

    # print("c1 shape : ", c1.shape)

    c2 = Conv2D(64, (1, 1), activation='relu', padding='same')(m)
    c2 = Conv2D(64, (7, 1), activation='relu', padding='same')(c2)
    c2 = Conv2D(64, (1, 7), activation='relu', padding='same')(c2)
    c2 = Conv2D(96, (3, 3), activation='relu', padding='valid')(c2)

    # print("c2 shape :", c2.shape)

    m2 = merge.concatenate([c1, c2], axis=channel_axis)

    # print("m2 shape :", m2.shape)
    # print("#########################")

    p1 = MaxPooling2D((3, 3), strides=(2, 2), )(m2)

    # print("p1 shape :", p1.shape)

    p2 = Conv2D(192, (3, 3), activation='relu', strides=(2, 2))(m2)

    # print("p1 shape :", p2.shape)

    m3 = merge.concatenate([p1, p2], axis=channel_axis)

    # print("m3 shape :", m3.shape)
    # print("#########################")

    m3 = BatchNormalization(axis=channel_axis)(m3)
    m3 = Activation('relu')(m3)
    return m3


def inception_resnet_v2_A(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input)

    ir2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input)
    ir2 = Conv2D(32, (3, 3), activation='relu', padding='same')(ir2)

    ir3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input)
    ir3 = Conv2D(48, (3, 3), activation='relu', padding='same')(ir3)
    ir3 = Conv2D(64, (3, 3), activation='relu', padding='same')(ir3)

    ir_merge = merge.concatenate([ir1, ir2, ir3], axis=channel_axis)

    # print("****************************************")
    # print("ir_merge shape :", ir_merge.shape)

    # print("Conv2D filters ", backend.int_shape(input)[channel_axis])
    ir_conv = Conv2D(backend.int_shape(input)[channel_axis], (1, 1), activation='relu')(ir_merge)
    out = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                 output_shape=backend.int_shape(input)[1:],
                 arguments={'scale': 0.1})([input, ir_conv])

    # out = merge.concatenate([init, ir_conv], axis=channel_axis)

    # print("out shape :", out.shape)

    # print("****************************************")

    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def inception_resnet_v2_B(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(192, (1, 1), activation='relu', padding='same')(input)

    ir2 = Conv2D(128, (1, 1), activation='relu', padding='same')(input)
    ir2 = Conv2D(160, (1, 7), activation='relu', padding='same')(ir2)
    ir2 = Conv2D(192, (7, 1), activation='relu', padding='same')(ir2)

    ir_merge = merge.concatenate([ir1, ir2], axis=channel_axis)

    # print("inception_resnet_v2_B   ir_merge : ", ir_merge.shape)

    ir_conv = Conv2D(backend.int_shape(input)[channel_axis], (1, 1), activation='relu')(ir_merge)
    out = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                 output_shape=backend.int_shape(input)[1:],
                 arguments={'scale': 0.1})([input, ir_conv])

    # ir_conv = Conv2D(1152, (1, 1), activation='linear', padding='same')(ir_merge)
    # if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)
    #
    # out = merge.concatenate([init, ir_conv], axis=channel_axis)

    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)

    # print("inception_resnet_v2_B   out : ", out.shape)

    return out


def inception_resnet_v2_C(input, scale_residual=True):
    channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Conv2D(192, (1, 1), activation='relu', padding='same')(input)

    ir2 = Conv2D(192, (1, 1), activation='relu', padding='same')(input)
    ir2 = Conv2D(224, (1, 3), activation='relu', padding='same')(ir2)
    ir2 = Conv2D(256, (3, 1), activation='relu', padding='same')(ir2)

    ir_merge = merge.concatenate([ir1, ir2], axis=channel_axis)

    ir_conv = Conv2D(backend.int_shape(input)[channel_axis], (1, 1), activation='relu')(ir_merge)
    out = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                 output_shape=backend.int_shape(input)[1:],
                 arguments={'scale': 0.1})([input, ir_conv])

    # ir_conv = Conv2D(2144, (1, 1), activation='linear', padding='same')(ir_merge)
    # if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)
    # out = merge.concatenate([init, ir_conv], axis=channel_axis)

    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def reduction_A(input, k=192, l=224, m=256, n=384):
    channel_axis = -1

    r1 = MaxPooling2D((3, 3), strides=(2, 2))(input)

    r2 = Conv2D(n, (3, 3), activation='relu', strides=(2, 2))(input)

    r3 = Conv2D(k, (1, 1), activation='relu', padding='same')(input)
    r3 = Conv2D(l, (3, 3), activation='relu', padding='same')(r3)
    r3 = Conv2D(m, (3, 3), activation='relu', strides=(2, 2))(r3)

    m = merge.concatenate([r1, r2, r3], axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m


def reduction_resnet_v2_B(input):
    channel_axis = -1

    r1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    r2 = Conv2D(256, (1, 1), activation='relu', padding='same')(input)
    r2 = Conv2D(384, (3, 3), activation='relu', strides=(2, 2))(r2)

    r3 = Conv2D(256, (1, 1), activation='relu', padding='same')(input)
    r3 = Conv2D(288, (3, 3), activation='relu', strides=(2, 2))(r3)

    r4 = Conv2D(256, (1, 1), activation='relu', padding='same')(input)
    r4 = Conv2D(288, (3, 3), activation='relu', padding='same')(r4)
    r4 = Conv2D(320, (3, 3), activation='relu', strides=(2, 2))(r4)

    m = merge.concatenate([r1, r2, r3, r4], axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m


def create_inception_resnet_v2(input_shape, reshape, nb_classes, scale=True):
    """
    create create_inception_resnet_v2 model
    :param input_shape:
    :param reshape:
    :param nb_classes:
    :param scale:
    :return:
    """

    # if K.image_dim_ordering() == 'th':
    #     init = Input((3, 299, 299))
    # else:
    # init = Input((input_shape))
    init = Input(shape=input_shape)
    if reshape:
        init = Reshape(reshape)(init)
    else:
        init = init

    # input_x = tf.pad(init, [[0, 0], [32, 32], [32, 32], [0, 0]])
    input_x = layers.ZeroPadding2D((32, 32))(init)

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(input_x)

    print("###### inception_resnet_stem : ", x.shape)

    # 10 x Inception Resnet A
    for i in range(10):
        x = inception_resnet_v2_A(x, scale_residual=scale)

    print("###### 10 x Inception Resnet A : ", x.shape)

    # Reduction A
    x = reduction_A(x, k=256, l=256, m=384, n=384)

    print("###### Reduction A : ", x.shape)

    # 20 x Inception Resnet B
    for i in range(20):
        x = inception_resnet_v2_B(x, scale_residual=scale)

    print("###### 20 x Inception Resnet B : ", x.shape)

    # Auxiliary tower
    aux_out = AveragePooling2D((5, 5), strides=3, padding='same')(x)
    aux_out = Conv2D(128, 1, 1, padding='same', activation='relu')(aux_out)
    aux_out = Conv2D(768, 5, 5, activation='relu', padding='same')(aux_out)
    aux_out = Flatten()(aux_out)
    aux_out = Dense(nb_classes, activation='softmax')(aux_out)

    # Reduction Resnet B
    x = reduction_resnet_v2_B(x)

    # 10 x Inception Resnet C
    for i in range(10):
        x = inception_resnet_v2_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling2D((8, 8), padding='same')(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(units=nb_classes, activation='softmax')(x)

    model = Model(inputs=init, outputs=out, name='model_Inception-Resnet-v2')

    return model


# model = create_inception_resnet_v2()
#
# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.summary()
#
# # if not set indent=4 ,it will out put Encodeing dict exception
# outputdf = sqlc.createDataFrame([Row(model_config=json.dumps(model.get_config(), cls=MyEncoder, indent=4))])


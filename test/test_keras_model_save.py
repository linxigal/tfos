# -*- coding: utf-8 -*-
"""
:Author  : weijinlong
:Time    : 
:File    : 
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import to_categorical
from tensorflow.contrib.learn.python.learn.datasets import mnist
from tensorflow.python.keras.callbacks import LambdaCallback, TensorBoard, ModelCheckpoint

batch_size = 100
steps_per_epoch = 10

with open('/Users/wjl/data/mnist/train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = np.array(mnist.extract_images(f))
with open('/Users/wjl/data/mnist/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = np.array(mnist.extract_labels(f, one_hot=True))

with open('/Users/wjl/data/mnist/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = np.array(mnist.extract_images(f))
with open('/Users/wjl/data/mnist/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = np.array(mnist.extract_labels(f, one_hot=True))

train_shape = train_images.shape
print("train_images.shape: {0}".format(train_shape))  # 60000 x 28 x 28
print("train_labels.shape: {0}".format(train_labels.shape))  # 60000 x 10
train_images = train_images.reshape(train_shape[0], train_shape[1] * train_shape[2])

test_shape = test_images.shape
print("test_images.shape: {0}".format(test_shape))  # 60000 x 28 x 28
print("test_labels.shape: {0}".format(test_labels.shape))  # 60000 x 10
test_images = test_images.reshape(test_shape[0], test_shape[1] * test_shape[2])


def data_generator(train):
    if train:
        max_batch_index = len(train_images) // batch_size
    else:
        max_batch_index = len(test_images) // batch_size
    i = 0
    while 1:
        if train:
            yield (
                train_images[i * batch_size: (i + 1) * batch_size], train_labels[i * batch_size: (i + 1) * batch_size])
        else:
            yield (test_images[i * batch_size: (i + 1) * batch_size], test_labels[i * batch_size: (i + 1) * batch_size])
        i += 1
        i = i % max_batch_index


def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        model_dir = 'saver_checkpoint'
        tensorboard_dir = 'tensorboard'
        model_checkpoint_dir = 'model_checkpoint/model_checkpoint'
        model_save_dir = "model_save"
        model_save_weight_dir = "model_save_weight"

        def save_checkpoint(epoch, logs=None):
            if epoch == 1:
                tf.train.write_graph(sess.graph.as_graph_def(), model_dir, 'graph.pbtxt')
            saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=epoch * steps_per_epoch)

        ckpt_callback = LambdaCallback(on_epoch_end=save_checkpoint)
        tb_callback = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(model_checkpoint_dir, monitor='loss', verbose=2, save_weights_only=False,
                                     mode='min')

        model.fit_generator(data_generator(True)
                            , steps_per_epoch=steps_per_epoch
                            , epochs=5
                            , verbose=1
                            , validation_data=data_generator(False)
                            , validation_steps=100
                            , callbacks=[
                ckpt_callback,
                tb_callback,
                # checkpoint
            ]
                            )
        model.save(model_save_dir)
        model.save_weights(model_save_weight_dir)


if __name__ == "__main__":
    build_model()

#!/usr/bin/env python
# -*- coding:utf-8 _*-  
"""
:Author :weijinlong
:Time:  :2019/7/19 10:02
:File   : cgan_classifier.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
# from .datas import mnist
# from .nets import *
from tensorflow.examples.tutorials.mnist import input_data
from tfos import ROOT_PATH


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# for test
def sample_y(m, n, ind):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, i % 8] = 1
    # y[:,7] = 1
    # y[-1,0] = 1
    return y


class CGAN_Classifier(object):
    def __init__(self, img_shape=(28, 28, 1), num_classes=10, z_dim=100, sample_dir=None, ckpt_dir='ckpt'):
        self.img_shape = img_shape
        self.img_row, self.img_col, self.channel = img_shape
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.sample_dir = sample_dir
        self.ckpt_dir = ckpt_dir

        self.X = tf.placeholder(tf.float32, shape=[None, self.img_row, self.img_col, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes])

        # nets
        self.G_sample = self.generator(tf.concat([self.z, self.y], 1))

        self.D_real, _ = self.discriminator(self.X)
        self.D_fake, _ = self.discriminator(self.G_sample, reuse=True)

        self.C_real = self.classifier(self.X)
        self.C_fake = self.classifier(self.G_sample, reuse=True)

        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(
            self.D_real))) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.C_real_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y))  # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))

        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5) \
            .minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5) \
            .minimize(self.G_loss + self.C_fake_loss, var_list=self.generator.vars)
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5) \
            .minimize(self.C_real_loss, var_list=self.classifier.vars)
        # self.C_fake_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
        # .minimize(self.C_fake_loss, var_list=self.generator.vars)

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def generator(self, x):
        with tf.variable_scope("generator") as scope:
            g = tcl.fully_connected(x, 7 * 7 * 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tf.reshape(g, (-1, 7, 7, 128))  # 7x7
            g = tcl.conv2d_transpose(g, 64, 4, stride=2,  # 14x14x64
                                     activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 1, 4, stride=2,  # 28x28x1
                                     activation_fn=tf.nn.sigmoid, padding='SAME',
                                     weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g

    def discriminator(self, x):
        with tf.variable_scope('discriminator') as scope:
            size = 64
            # bzx28x28x1 -> bzx14x14x64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=5, stride=2, activation_fn=lrelu)
            # 7x7x128
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=5, stride=2, activation_fn=lrelu,
                                normalizer_fn=tcl.batch_norm)
            shared = tcl.flatten(shared)
            d = tcl.fully_connected(shared, 1, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, self.num_classes, activation_fn=None)  # 10 classes
            return d, q

    def classifier(self, x):
        with tf.variable_scope('classifier') as scope:
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=5,  # bzx28x28x1 -> bzx14x14x64
                                stride=2, activation_fn=tf.nn.relu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=5,  # 7x7x128
                                stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.fully_connected(tcl.flatten(  # reshape, 1
                shared), 1024, activation_fn=tf.nn.relu)

            # c = tcl.fully_connected(shared, 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            c = tcl.fully_connected(shared, self.num_classes, activation_fn=None)  # 10 classes
            return c

    def train(self, epochs=1000, batch_size=32, sample_interval=200):
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            # update D
            for _ in range(1):
                X_b, y_b = data.train.next_batch(batch_size)
                X_b = tf.reshape(X_b, self.img_shape)
                self.sess.run(self.D_solver,
                              feed_dict={self.X: X_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
            # update G
            for _ in range(1):
                self.sess.run(self.G_solver, feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
            # update C
            for _ in range(1):
                # real label to train C
                self.sess.run(self.C_real_solver, feed_dict={self.X: X_b, self.y: y_b})
            '''
                # fake img label to train G
                self.sess.run(
                    self.C_fake_solver,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
            '''
            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                d_loss_curr, c_real_loss_curr = self.sess.run([self.D_loss, self.C_real_loss],
                                                              feed_dict={self.X: X_b, self.y: y_b,
                                                                         self.z: sample_z(batch_size, self.z_dim)})
                g_loss_curr, c_fake_loss_curr = self.sess.run([self.G_loss, self.C_fake_loss],
                                                              feed_dict={self.y: y_b,
                                                                         self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:.4}; C_fake_loss: {:.4}'
                      .format(epoch, d_loss_curr, g_loss_curr, c_real_loss_curr, c_fake_loss_curr))

                if epoch % 1000 == 0:
                    y_s = sample_y(16, self.num_classes, fig_count % 10)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    # fig = self.data.data2fig(samples)
                    # plt.savefig('{}/{}_{}.png'.format(self.sample_dir, str(fig_count).zfill(3), str(fig_count % 10)),
                    #             bbox_inches='tight')
                    fig_count += 1
                    # plt.close(fig)

        # if epoch % 2000 == 0:
        #	self.saver.save(self.sess, os.path.join(ckpt_dir, "cgan_classifier.ckpt"))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    #
    # # save generated images
    # sample_dir = 'Samples/mnist_cgan_classifier'
    # if not os.path.exists(sample_dir):
    #     os.makedirs(sample_dir)
    #
    # # param
    # generator = G_conv_mnist()
    # discriminator = D_conv_mnist()
    # classifier = C_conv_mnist()
    #
    # data = mnist()
    #
    # # run
    # cgan_c = CGAN_Classifier(generator, discriminator, classifier, data)
    # cgan_c.train(sample_dir)
    data_path = os.path.join(ROOT_PATH, 'data', 'mnist')

    data = input_data.read_data_sets(data_path, one_hot=True)
    CGAN_Classifier().train(data)

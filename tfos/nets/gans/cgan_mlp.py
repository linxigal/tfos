#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
:Author     : weijinlong
:Time:      : 2019/7/22 11:18
:File       : image_dir.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# for test
def sample_y(m, n, ind):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, ind] = 1
    return y


def concat(z, y):
    return tf.concat([z, y], 1)


class CGAN_MLP(object):
    def __init__(self, data, output_dir, ckpt_dir):
        self.data = data
        self.output_dir = output_dir
        self.ckpt_dir = ckpt_dir

        # data
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim  # condition
        self.X_dim = self.data.X_dim

        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        # nets
        self.G_sample = self.generator(concat(self.z, self.y))
        self.D_real, _ = self.discriminator(concat(self.X, self.y))
        self.D_fake, _ = self.discriminator(concat(self.G_sample, self.y), reuse=True)

        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(
            self.D_real))) + tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # solver
        self.D_solver = tf.train.AdamOptimizer().minimize(
            self.D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
        self.G_solver = tf.train.AdamOptimizer().minimize(
            self.G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def generator(self, z):
        with tf.variable_scope('generator') as vs:
            g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
        return g

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu,
                                         weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))

            q = tcl.fully_connected(shared, self.y_dim, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))  # 10 classes

        return d, q

    def train(self, steps, batch_size):
        fig_count = 0
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(steps):
            # update D
            x_b, y_b = self.data(batch_size)
            self.sess.run(
                self.D_solver,
                feed_dict={self.X: x_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)}
            )
            # update G
            k = 1
            for _ in range(k):
                self.sess.run(
                    self.G_solver,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)}
                )

            # save img, model. print loss
            if epoch % 100 == 0 or epoch < 100:
                d_loss_curr = self.sess.run(
                    self.D_loss,
                    feed_dict={self.X: x_b, self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                g_loss_curr = self.sess.run(
                    self.G_loss,
                    feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(epoch, d_loss_curr, g_loss_curr))

                if epoch % 1000 == 0:
                    y_s = sample_y(16, self.y_dim, fig_count % 10)
                    samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})

                    fig = self.data.data2fig(samples)
                    plt.savefig('{}/{}_{}.png'.format(self.output_dir, str(fig_count).zfill(3), str(fig_count % 10)),
                                bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

            if epoch % 2000 == 0:
                self.saver.save(self.sess, os.path.join(self.ckpt_dir, "cgan.ckpt"))

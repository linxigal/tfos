#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
:Author     : weijinlong
:Time:      : 2019/7/22 11:18
:File       : image_dir.py
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflowonspark import TFCluster, TFNode

from tfos.utils.file_manager import HDFSOP


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
    def __init__(self, data, output_path, ckpt_path, steps, batch_size):
        self.data = data
        self.output_path = output_path
        self.ckpt_path = ckpt_path
        self.steps = steps
        self.batch_size = batch_size

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
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step = tf.train.get_or_create_global_step()
        self.D_solver = tf.train.AdamOptimizer().minimize(
            self.D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'),
            global_step=self.global_step)
        self.G_solver = tf.train.AdamOptimizer().minimize(
            self.G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

        self.saver = tf.train.Saver(max_to_keep=3)
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

    def train(self):
        fig_count = 0
        steps = self.steps
        batch_size = self.batch_size

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(1, steps + 1):
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
                    image_path = '{}/{}_{}.png'.format(self.output_path, str(fig_count).zfill(3), str(fig_count % 10))
                    self.save_image(image_path)
                    # plt.savefig('{}/{}_{}.png'.format(self.output_path, str(fig_count).zfill(3), str(fig_count % 10)),
                    #             bbox_inches='tight')
                    fig_count += 1
                    plt.close(fig)

            if epoch % 2000 == 0:
                self.saver.save(self.sess, os.path.join(self.ckpt_path, "cgan.ckpt"), global_step=self.global_step)

    def save_image(self, image_path):
        if 'hdfs' in self.output_path:
            buf = io.BytesIO()
            plt.savefig(buf, bbox_inches='tight')
            buf.seek(0)
            with HDFSOP.write(image_path, overwrite=True) as write:
                write.write(buf.getvalue())
            buf.close()
        else:
            plt.savefig(image_path, bbox_inches='tight')

    def __call__(self, args, ctx):
        self.task_index = ctx.task_index
        self.job_name = ctx.job_name
        self.cluster, self.server = TFNode.start_cluster_server(ctx)
        self.tf_feed = TFNode.DataFeed(ctx.mgr)

        if ctx.job_name == "ps":
            self.server.join()
        elif ctx.job_name == "worker":
            self.train()


class TFOS_CGAN_MLP(object):
    def __init__(self, sc, cluster_size, num_ps, input_mode=TFCluster.InputMode.TENSORFLOW):
        self.sc = sc
        self.input_mode = input_mode
        self.cluster_size = cluster_size
        self.num_ps = num_ps

    def train(self, data, output_path, steps, batch_size):
        checkpoint_path = os.path.join(output_path, 'checkpoint')
        if not tf.gfile.Exists(checkpoint_path):
            tf.gfile.MkDir(checkpoint_path)
        result_path = os.path.join(output_path, 'results')
        if not tf.gfile.Exists(result_path):
            tf.gfile.MkDir(result_path)
        worker = CGAN_MLP(data, result_path, checkpoint_path, steps, batch_size)
        cluster = TFCluster.run(self.sc, worker, None, self.cluster_size, self.num_ps, input_mode=self.input_mode)
        cluster.shutdown()

    @staticmethod
    def local_train(data, output_path, steps, batch_size):
        checkpoint_path = os.path.join(output_path, 'checkpoint')
        if tf.gfile.Exists(checkpoint_path):
            tf.gfile.DeleteRecursively(checkpoint_path)
        tf.gfile.MkDir(checkpoint_path)
        result_path = os.path.join(output_path, 'results')
        if tf.gfile.Exists(result_path):
            tf.gfile.DeleteRecursively(result_path)
        tf.gfile.MkDir(result_path)
        CGAN_MLP(data, result_path, checkpoint_path, steps, batch_size).train()

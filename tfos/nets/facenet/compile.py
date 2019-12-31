#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2020/1/21 9:06
:File   :compile.py
:content:
  
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from facenet.src import facenet

from tfos.tf.compile import TFCompile


class FaceNetSoftMaxCompile(TFCompile):

    def __init__(self, model_rdd,
                 prelogits_norm_p=1.0,
                 prelogits_norm_loss_factor=0.0,
                 center_loss_alfa=0.95,
                 center_loss_factor=0.0,
                 learning_rate_decay_epochs=100,
                 learning_rate_decay_factor=1.0,
                 optimizer='ADAGRAD',
                 moving_average_decay=0.9999,
                 log_histograms=False):
        super(FaceNetSoftMaxCompile, self).__init__()
        self.deserialize(model_rdd)
        self.weight_decay = self.params['weight_decay']
        self.n_classes = self.params['n_classes']
        self.prelogits_norm_p = prelogits_norm_p
        self.prelogits_norm_loss_factor = prelogits_norm_loss_factor
        self.center_loss_alfa = center_loss_alfa
        self.center_loss_factor = center_loss_factor
        self.learning_rate_decay_epochs = learning_rate_decay_epochs
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.optimizer = optimizer
        self.moving_average_decay = moving_average_decay
        self.log_histograms = log_histograms

    def compile(self):
        _, outputs = self.outputs
        prelogits, label_batch = outputs
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        steps_per_epoch_placeholder = tf.placeholder(tf.int32, name='steps_per_epoch')
        n_classes_placeholder = tf.placeholder(tf.int32, name='n_classes')
        self.add_inputs(learning_rate_placeholder, steps_per_epoch_placeholder, n_classes_placeholder)
        logits = slim.fully_connected(prelogits, self.n_classes, activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(self.weight_decay),
                                      scope='Logits', reuse=False)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=self.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * self.prelogits_norm_loss_factor)

        # Add center loss
        prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, self.center_loss_alfa, self.n_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * self.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, self.global_step,
                                                   self.learning_rate_decay_epochs * steps_per_epoch_placeholder,
                                                   self.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, self.global_step, self.optimizer,
                                 learning_rate, self.moving_average_decay, tf.global_variables(),
                                 self.log_histograms)

        self.add_compiles(total_loss,
                          train_op,
                          # regularization_losses,
                          logits,
                          cross_entropy_mean,
                          learning_rate,
                          prelogits_norm,
                          accuracy,
                          prelogits_center_loss)
        return self

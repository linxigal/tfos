#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/10 11:28
:File   :recurrent.py
:content:
  
"""

from tensorflow.python.keras.layers import SimpleRNN, GRU, LSTM

from tfos.base import BaseLayer, ext_exception


class SimpleRNNLayer(BaseLayer):
    __doc__ = SimpleRNN.__doc__

    @ext_exception("SimpleRNN Layer")
    def add(self,
            units,
            activation='tanh',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.,
            recurrent_dropout=0.,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            **kwargs):
        return self._add_layer(SimpleRNN(units=units,
                                         activation=activation,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         recurrent_initializer=recurrent_initializer,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         recurrent_regularizer=recurrent_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint,
                                         recurrent_constraint=recurrent_constraint,
                                         bias_constraint=bias_constraint,
                                         dropout=dropout,
                                         recurrent_dropout=recurrent_dropout,
                                         return_sequences=return_sequences,
                                         return_state=return_state,
                                         go_backwards=go_backwards,
                                         stateful=stateful,
                                         unroll=unroll,
                                         **kwargs))


class GRULayer(BaseLayer):
    __doc__ = GRU.__doc__

    @ext_exception("GRU Layer")
    def add(self,
            units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.,
            recurrent_dropout=0.,
            implementation=1,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            reset_after=False,
            **kwargs):
        return self._add_layer(GRU(units=units,
                                   activation=activation,
                                   recurrent_activation=recurrent_activation,
                                   use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   recurrent_initializer=recurrent_initializer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   recurrent_regularizer=recurrent_regularizer,
                                   bias_regularizer=bias_regularizer,
                                   activity_regularizer=activity_regularizer,
                                   kernel_constraint=kernel_constraint,
                                   recurrent_constraint=recurrent_constraint,
                                   bias_constraint=bias_constraint,
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   implementation=implementation,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   reset_after=reset_after,
                                   **kwargs))


class LSTMLayer(BaseLayer):
    __doc__ = LSTM.__doc__

    @ext_exception("LSTMLayer Layer")
    def add(self,
            units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.,
            recurrent_dropout=0.,
            implementation=1,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            **kwargs):
        return self._add_layer(LSTM(units=units,
                                    activation=activation,
                                    recurrent_activation=recurrent_activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    unit_forget_bias=unit_forget_bias,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    recurrent_constraint=recurrent_constraint,
                                    bias_constraint=bias_constraint,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout,
                                    implementation=implementation,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    go_backwards=go_backwards,
                                    stateful=stateful,
                                    unroll=unroll,
                                    **kwargs))

#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/10/14 17:00
:File   :embeddings.py
:content:
  
"""

from tensorflow.python.keras.layers import Embedding

from tfos.base import BaseLayer, ext_exception


class EmbeddingLayer(BaseLayer):
    __doc__ = Embedding.__doc__

    @ext_exception('Embedding layer')
    def add(self, input_dim, output_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
            **kwargs):
        return self._add_layer(Embedding(input_dim=input_dim,
                                         output_dim=output_dim,
                                         embeddings_initializer=embeddings_initializer,
                                         embeddings_regularizer=embeddings_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         embeddings_constraint=embeddings_constraint,
                                         mask_zero=mask_zero,
                                         input_length=input_length, **kwargs))

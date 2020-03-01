# -*- coding:utf-8 _*-

from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.python.keras.models import Sequential
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import ConditionalRandomFieldLoss

from tfos.base import *


class BilstmCrf(BaseLayer):

    @ext_exception('BilstmCrf model')
    def add(self, vocab_size, EMBED_DIM, BiRNN_UNITS, tags_size):
        model = create_bilstm_crf(vocab_size, EMBED_DIM, BiRNN_UNITS, tags_size)
        return self.model2df(model)


def create_bilstm_crf(vocab_size, EMBED_DIM, BiRNN_UNITS, tags_size):

    model = Sequential()
    # model.add(Embedding(len(vocab)+1, EMBED_DIM, mask_zero=True))
    model.add(Embedding(vocab_size, EMBED_DIM, mask_zero=True))
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    model.add(CRF(tags_size, sparse_target=True, name="crf_layer"))
    print(model.summary())

    crf_loss_instance = ConditionalRandomFieldLoss()
    # model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    #model.compile('adam', loss={"crf_layer": crf_loss_instance}, metrics=[crf_accuracy])

    model.summary()

    return model

#!/usr/bin/env python
# -*- coding:utf-8 _*-

"""
:Author :weijinlong
:Time:  :2019/9/25 8:45
:File   :cifar.py
:content:

"""

import platform
from collections import Counter

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from tfos.data import BaseData


class CrfWord2id(BaseData):
    def __init__(self, **kwargs):
        super(CrfWord2id, self).__init__(**kwargs)

    @property
    def train_df(self):
        rdd = self.sc.parallelize(zip(*self.load_train()))
        return self.rdd2df(rdd)

    def load_train(self):

        print("$$$$$$$$$$$$$$$$$")
        fh = open(self.path, 'rb')

        if platform.system() != 'Windows':
            split_text = '\r\n'
        else:
            split_text = '\n'

        string = fh.read().decode('utf-8')
        data = [[row.split() for row in sample.split(split_text)] for
                sample in
                string.strip().split(split_text + split_text)]
        fh.close()

        word_counts = Counter(row[0].lower() for sample in data for row in sample)
        # print(word_counts)
        vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
        print('###### vocab size : {} #######'.format(len(vocab)))
        chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

        print("-------len chunk_tags size : {}-------------".format(len(chunk_tags)))

        # train = _process_data(train, vocab, chunk_tags, maxlen=100)
        # test = _process_data(test, vocab, chunk_tags, maxlen=100)

        word2idx = dict((w, i) for i, w in enumerate(vocab))

        # print(word2idx)

        x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

        # print(x)

        y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
        # print(y_chunk)

        # print("222222222222222222222222")

        x = pad_sequences(x, 100, padding='post')  # right padding
        # y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
        y_chunk = pad_sequences(y_chunk, 100, value=0, padding='post')

        return x.tolist(), y_chunk.tolist()

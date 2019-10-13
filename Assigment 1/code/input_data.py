#encoding:utf-8

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

import importlib,sys
importlib.reload(sys)

import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import itertools


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, "input.zh.txt")
        vocab_file = os.path.join(data_dir, "vocab.zh.pkl")

        self.preprocess(input_file, vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()
        if not isinstance(sentences, list): #判断对象是否是已知的类型
            sentences = [sentences]
        for sent in sentences:      #添加元素
            word_counts.update(sent)
        print(word_counts.most_common())
        """
        将所有词频大于3的作为词汇表，总共有4557个
        """
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq] #most_common()未指明N则列出所有基数，选择大于mini_frq的
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}   #顺序变化一下，i x变成x i
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8: #有无BOM标记
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]
        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)
        print(len(self.vocab))
        print(self.vocab)
        #print 'word num: ', self.vocab_size

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)

        #将每一行表示成[0 0 0 0 0 .... 2 2 2 2 2]，,,,表示每个单词在vocab中的序号，不出现用1代替
        raw_data = [[0] * self.seq_length +
            [self.vocab.get(w, 1) for w in line] +
            [2] * self.seq_length for line in lines]

        self.raw_data = raw_data

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data:
            """
            w1,w2,w3,w4,w5作为x；w6作为y
            w2,w3,w4,w5,w6作为x；w7作为y
            ...
            ...
            """
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind-self.seq_length:ind])
                ydata.append([row[ind]])
        self.num_batches = int(len(xdata) / self.batch_size)
        print(self.num_batches )
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        #进行切分
        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


if __name__=="__main__":
    data=TextLoader('data/',120,5)


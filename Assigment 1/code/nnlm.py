# -*- coding: utf-8 -
import importlib,sys
importlib.reload(sys)

#encoding:utf-8
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')

from input_data import *

import numpy as np
import tensorflow as tf
import argparse
import time
import math
from tensorflow.python.platform import gfile

def main():
    #参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='data directory containing input.txt')
    parser.add_argument('--batch_size', type=int, default=120,
                       help='minibatch size')
    parser.add_argument('--win_size', type=int, default=5,
                       help='context sequence length')  #上下文序列的长度
    parser.add_argument('--hidden_num', type=int, default=64,
                       help='number of hidden layers')  #隐藏层的个数
    parser.add_argument('--word_dim', type=int, default=50,
                       help='number of word embedding') #词向量维度
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='number of epochs') #迭代的次数
    parser.add_argument('--grad_clip', type=float, default=10.,
                       help='clip gradients at this value') #防止梯度爆炸，不能让梯度超过10

    args = parser.parse_args() #参数集合

    #准备训练数据
    data_loader = TextLoader(args.data_dir, args.batch_size, args.win_size)
    args.vocab_size = data_loader.vocab_size    #词汇表大小
	
    #模型定义
    graph = tf.Graph()
    with graph.as_default():
        #定义训练数据
        input_data = tf.placeholder(tf.int32, [args.batch_size, args.win_size])     #120*5
        targets = tf.placeholder(tf.int64, [args.batch_size, 1])    #120*1
		
        #模型参数
        """
        将每个词转化成词向量C,利用tf.nn.embedding_lookup
        """
        with tf.variable_scope('nnlm' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))  #产生值在-1到1之间的vocab_size*word_size的矩阵 4557*50
            embeddings = tf.nn.l2_normalize(embeddings, 1)  #按行进行l2范化

        with tf.variable_scope('nnlm' + 'weight'):
            #tf.truncated_normal产生满足正态分布的数据
            weight_h = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim + 1, args.hidden_num],  #[5*50+1,64]
                            stddev=1.0 / math.sqrt(args.hidden_num)))
            softmax_w = tf.Variable(tf.truncated_normal([args.win_size * args.word_dim, args.vocab_size],     #[250,4557]
                            stddev=1.0 / math.sqrt(args.win_size * args.word_dim)))
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num + 1, args.vocab_size],              #[65*4557]
                            stddev=1.0 / math.sqrt(args.hidden_num)))


        #TODO，构造计算图
        def infer_output(input_data):
            # step 1: hidden = tanh(x * H + d)
			# step 2: outputs = softmax(x * W + hidden * U + b)
            X_input=tf.nn.embedding_lookup(embeddings,input_data)  #输出数据的维度为120*5*50
            X_input=tf.reshape(X_input,[-1,args.win_size*args.word_dim]) #转化成120*250
            b=tf.stack([tf.shape(input_data)[0],1])
            b=tf.ones(b)  #[120*1]
            X=tf.concat([X_input,b],1) #[120*251]

            hidden=tf.nn.tanh(tf.matmul(X,weight_h))  #[120*251]*[251*64]=[120*64]
            hidden=tf.concat([hidden,b],1)            #[120*65]
            outputs=tf.matmul(X_input,softmax_w)+tf.matmul(hidden,softmax_u) #[120*250]*[250*4557]+[120*65]*[65*4557]=[120*4557]
            outputs=tf.clip_by_value(outputs,0.0,args.grad_clip)  #将output中的每一个元素压缩到0-args.grad_clip范围内
            outputs=tf.nn.softmax(outputs)
            return outputs

        outputs = infer_output(input_data)
        one_hot_targets = tf.one_hot(tf.squeeze(targets), args.vocab_size, 1.0, 0.0) #one-hot编码

        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1))
        optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        #输出词向量
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm    #按照行的维度求和

    #模型训练
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {input_data: x, targets: y}
                train_loss,  _ = sess.run([loss, optimizer], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))
			# 保存词向量至nnlm_word_embeddings.npy文件
            np.save('nnlm_word_embeddings.zh', normalized_embeddings.eval())
        

if __name__ == '__main__':
    main()

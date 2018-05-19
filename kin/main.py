# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess
from keras.models import Model
from keras.optimizers import Adam, Nadam
from keras.layers import SpatialDropout1D, Bidirectional, CuDNNGRU, Conv1D, Embedding, Input, CuDNNLSTM, Dropout
from keras.layers import Layer, Lambda, PReLU, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Masking, GRU, Dense
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras import regularizers
from keras import backend as K
from math import ceil

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        data1, data2 = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(model_output, feed_dict={x1: data1, x2: data2})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]

def index_split(data_len, validation_rate=0.3):
    from math import ceil
    from random import shuffle
    train_len = ceil((1 - validation_rate) * data_len)
    validation_len = data_len - train_len

    total_index = [i for i in range(data_len)]
    shuffle(total_index)

    return total_index[:train_len], total_index[train_len:]

def _batch_loader2(iterable, index, n=1):
    length = len(index)
    
    for n_idx in range(0, length, n):
        yield iterable[index[n_idx:min(n_idx + n, length)]]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=1500)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=250)
    args.add_argument('--embedding', type=int, default=128)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--valrate', type=float, default=0.3)
    config = args.parse_args()

    sess = tf.InteractiveSession()
    K.set_session(sess)

    if config.mode == 'train':
        K.set_learning_phase(1)  # set learning phase
        dropout_rate = 0.2
    else:
        K.set_learning_phase(0)  # set learning phase
        dropout_rate = 0

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = 'sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding * config.strmaxlen
    learning_rate = 0.001
    character_size = 251

    x1 = Input(shape=(config.strmaxlen,), dtype='int32')
    x2 = Input(shape=(config.strmaxlen,), dtype='int32')
    y_ = Input(shape=(config.output,), dtype='float32')

    class ManhattanDistance(Layer):
        def __init__(self, **kwargs):
            self.result = None
            super(ManhattanDistance, self).__init__(**kwargs)

        def build(self, input_shape):
            super(ManhattanDistance, self).build(input_shape)

        def call(self, x, **kwargs):
            self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
            return self.result

        def compute_output_shape(self, input_shape):
            return K.int_shape(self.result)


    def sentence_embedding(x):
        with tf.variable_scope('sentence', reuse=tf.AUTO_REUSE):
            output = Embedding(character_size, config.embedding, mask_zero=True)(x)
            from padding import ZeroMaskedEntries
            zeroMask = ZeroMaskedEntries()
            output = zeroMask(output)
            output = SpatialDropout1D(dropout_rate)(output)
            output = Bidirectional(CuDNNGRU(config.embedding, return_sequences=True))(output)
            output = Conv1D(config.embedding, kernel_size=7, padding='same', kernel_initializer='glorot_uniform')(output)
            output = PReLU()(output)
            avg_pool = GlobalAveragePooling1D()(output)
            max_pool = GlobalMaxPooling1D()(output)
            from Attention import Attention
            att = Attention(config.strmaxlen, user_mask=zeroMask.get_mask())(output)
            output = concatenate([avg_pool, max_pool, att])
            output = Dropout(dropout_rate)(output)
        return output

    def word_cnn(x):
        filter_sizes = [1, 2, 3, 5, 7]
        num_filters = 32

        x = Embedding(character_size, config.embedding)(x)
        x = SpatialDropout1D(0.4)(x)
        x = Reshape((config.strmaxlen, config.embedding, 1))(x)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], config.embedding), kernel_initializer='normal',
                        activation='elu')(x)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], config.embedding), kernel_initializer='normal',
                        activation='elu')(x)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], config.embedding), kernel_initializer='normal',
                        activation='elu')(x)
        conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], config.embedding), kernel_initializer='normal',
                        activation='elu')(x)
        conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[4], config.embedding), kernel_initializer='normal',
                        activation='elu')(x)

        maxpool_0 = MaxPool2D(pool_size=(config.strmaxlen - filter_sizes[0] + 1, 1))(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(config.strmaxlen - filter_sizes[1] + 1, 1))(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(config.strmaxlen - filter_sizes[2] + 1, 1))(conv_2)
        maxpool_3 = MaxPool2D(pool_size=(config.strmaxlen - filter_sizes[3] + 1, 1))(conv_3)
        maxpool_4 = MaxPool2D(pool_size=(config.strmaxlen - filter_sizes[4] + 1, 1))(conv_4)

        z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
        z = Flatten()(z)
        z = Dropout(0.1)(z)
        return z

    sentence1 = sentence_embedding(x1)
    sentence2 = sentence_embedding(x2)
    # sentence1 = word_cnn(x1)
    # sentence2 = word_cnn(x2)

    manhattan_distance = ManhattanDistance()([sentence1, sentence2])
    model_output = manhattan_distance
    model = Model(inputs=[x1, x2], outputs=[manhattan_distance])

    def custom_loss(y_true, y_pred):
        positive_loss = y_true * (K.square(1. - y_pred))
        negative_loss = (1. - y_true) * K.square(y_pred)
        contrastive_loss = positive_loss + negative_loss

        margin = 0.25
        target_zero = tf.equal(y_true, 0.)
        less_than_margin = tf.less(y_pred, margin)
        both_logical = tf.logical_and(target_zero, less_than_margin)
        both_logical = tf.cast(both_logical, tf.float32)
        multiplicative_factor = tf.cast(1. - both_logical, tf.float32)
        contrastive_loss = tf.multiply(contrastive_loss, multiplicative_factor)
        loss_op = K.mean(contrastive_loss)
        return loss_op

    model.compile(loss=custom_loss, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001),
                  metrics=['accuracy'])
    model.summary()

    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        #one_batch_size = dataset_len//config.batch
        #if dataset_len % config.batch != 0:
        #    one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            train_avg_loss = 0.0
            train_avg_acc = 0.0
            validation_avg_loss = 0.0
            validation_avg_acc = 0.0

            train_index, validation_index = index_split(dataset_len, validation_rate=config.valrate)

            train_batch_size = ceil(len(train_index) / config.batch)
            validation_batch_size = ceil(len(validation_index) / config.batch)
            for i, (data1, data2, labels) in enumerate(_batch_loader2(dataset, train_index, config.batch)):
                loss, acc = model.train_on_batch([data1, data2], labels)
                train_avg_loss += float(loss)
                train_avg_acc += float(acc)

            for i, (data1, data2, labels) in enumerate(_batch_loader2(dataset, validation_index, config.batch)):
                loss, acc = model.test_on_batch([data1, data2], labels)
                validation_avg_loss += float(loss)
                validation_avg_acc += float(acc)

            if validation_batch_size > 0:
                print('epoch:', epoch,
                    ' train_loss:', float(train_avg_loss / train_batch_size),
                    ' train_accuracy:', float(train_avg_acc / train_batch_size),
                    ' validation_loss:', float(validation_avg_loss / validation_batch_size),
                    ' validation_accuracy:', float(validation_avg_acc / validation_batch_size),
                    ' lr:', model.optimizer.lr.eval())
                nsml.report(summary=True, scope=locals(),
                    epoch=epoch, epoch_total=config.epochs, step=epoch,
                    train_loss=float(train_avg_loss / train_batch_size),
                    train_accuracy=float(train_avg_acc / train_batch_size),
                    validation_loss=float(validation_avg_loss / validation_batch_size),
                    validation_accuracy=float(validation_avg_acc / validation_batch_size))
            else:
                print('epoch:', epoch,
                      ' train_loss:', float(train_avg_loss / train_batch_size),
                      ' train_accuracy:', float(train_avg_acc / train_batch_size))
                nsml.report(summary=True, scope=locals(),
                            epoch=epoch, epoch_total=config.epochs, step=epoch,
                            train_loss=float(train_avg_loss / train_batch_size),
                            train_accuracy=float(train_avg_acc / train_batch_size))
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        print(len(queries))
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(len(res))

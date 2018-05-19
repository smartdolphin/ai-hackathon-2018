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
import time

import numpy as np
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import nsml
from dataset import MovieReviewDataset, preprocess, group_count
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

from models import Regression

USE_GPU = torch.cuda.device_count()
if USE_GPU:
    print("using %d GPUs" % USE_GPU)

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data, data_lengths, char_ids, char_lengths, word_ids, word_lengths = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data, data_lengths, char_ids, char_lengths, word_ids, word_lengths)
        output_prediction = torch.clamp(output_prediction, 1., 10.)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    length = []
    review_char_id = []
    review_char_length = []
    word_id = []
    word_length = []
    label = []
    for datum in data:
        review.append(datum[0])
        length.append(datum[1])
        review_char_id.append(datum[2])
        review_char_length.append(datum[3])
        word_id.append(datum[4])
        word_length.append(datum[5])
        label.append(datum[6])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(length), np.array(review_char_id), np.array(review_char_length), \
            np.array(word_id), np.array(word_length), np.array(label)


def sorted_in_decreasing_order(data, lengths):
    sorted_index = np.argsort(-lengths)
    sorted_data = list(np.array(data)[sorted_index])
    sorted_lengths = lengths[sorted_index]
    return sorted_data, np.maximum(sorted_lengths, 1)


if __name__ == '__main__':
    print('torch version:', torch.__version__)
    print('numpy version:', np.__version__)
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=1000)
    args.add_argument('--strmaxlen', type=int, default=80) # 99.99% = 80, 100% = 117
    args.add_argument('--embedding', type=int, default=64)
    args.add_argument('--dropout', type=float, default=0.3)
    args.add_argument('--rnn_layers', type=int, default=2)
    args.add_argument('--max_dataset', type=int, default=-1)
    args.add_argument('--model_type', type=str, default='last')
    args.add_argument('--dataset_path', type=str)
    args.add_argument('--no_eval', action='store_true')
    config = args.parse_args()

    if config.mode == 'train':
        dropout_prob = config.dropout
        is_training = True
    else:
        dropout_prob = 0.0
        is_training = False

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        if config.dataset_path:
            DATASET_PATH = config.dataset_path
        else:
            DATASET_PATH = 'sample_data/movie_review/'

    model = Regression(config.embedding, config.strmaxlen, dropout_prob,
            config.rnn_layers, use_gpu=USE_GPU, model_type=config.model_type)
    if USE_GPU:
        #if USE_GPU > 1:
        #    model = nn.DataParallel(model)
        model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        t0 = time.time()
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen, max_size=config.max_dataset)
        print("dataset loaded %.2f s" % (time.time() - t0))
        pin_memory = USE_GPU > 0
        if config.no_eval:
            train_loader = DataLoader(dataset=dataset, batch_size=config.batch,
                                      shuffle=True, collate_fn=collate_fn,
                                      num_workers=2, pin_memory=pin_memory)
            eval_loader = []
        else:
            train_sampler, eval_sampler = dataset.get_sampler()
            train_loader = DataLoader(dataset=dataset, batch_size=config.batch,
                                      sampler=train_sampler, collate_fn=collate_fn,
                                      num_workers=2, pin_memory=pin_memory)
            eval_loader = DataLoader(dataset=dataset, batch_size=config.batch,
                                      sampler=eval_sampler, collate_fn=collate_fn,
                                      num_workers=2, pin_memory=pin_memory)
        total_batch = len(train_loader)
        total_eval_batch = min(len(eval_loader), 30)
        print("total batch:", total_batch)
        print("total eval batch:", total_eval_batch)

        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_accuracy = 0.0
            t0 = time.time()
            t1 = time.time()
            for i, (data, lengths, char_ids, char_lengths, word_ids, word_lengths, labels) in enumerate(train_loader):
                # 아래코드 때문에 학습이 제대로 안된다. 알 수 없음
                #data, lengths = sorted_in_decreasing_order(data, lengths)

                # 1, 10은 멀리 떨어트리기
                #labels[labels < 2] = 0.5
                #labels[labels > 9] = 10.5

                predictions = model(data, lengths, char_ids, char_lengths, word_ids, word_lengths)
                label_vars = Variable(torch.from_numpy(labels))
                if USE_GPU:
                    label_vars = label_vars.cuda()
                loss = criterion(predictions, label_vars)
                if USE_GPU:
                    loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                predictions = torch.clamp(predictions, 1., 10.)
                label_vars = torch.clamp(label_vars, 1., 10.)
                correct = label_vars.eq(torch.round(predictions.view(-1)))
                accuracy = correct.float().mean().data[0]
                avg_loss += loss.data[0]
                avg_accuracy += accuracy

                if (i+1) % 200 == 0:
                    print('Batch : ', i + 1, '/', total_batch,
                          ', MSE in this minibatch: ', round(loss.data[0], 3),
                          ', Accuracy:', round(accuracy, 2), ', time: %.2f' % (time.time() - t0))
                    t0 = time.time()

            print(group_count('label', labels))
            print(group_count('prediction', torch.round(predictions.view(-1)).data.tolist()))

            train_accuracy = float(avg_accuracy / total_batch)
            train_loss = float(avg_loss/total_batch)
            print('epoch:', epoch, 'train_loss: %.3f' % train_loss, 'accuracy: %.2f' % train_accuracy,
                    ', time: %.1fs' % (time.time() - t1))

            # Evaluation
            avg_loss = 0.0
            avg_accuracy = 0.0
            t0 = time.time()
            t1 = time.time()
            for i, (data, lengths, char_ids, char_lengths, word_ids, word_lengths, labels) in enumerate(eval_loader):
                #data, lengths = sorted_in_decreasing_order(data, lengths)

                predictions = model(data, lengths, char_ids, char_lengths, word_ids, word_lengths)
                predictions = torch.clamp(predictions, 1., 10.)
                label_vars = Variable(torch.from_numpy(labels))
                if USE_GPU:
                    label_vars = label_vars.cuda()
                loss = criterion(predictions, label_vars)
                if USE_GPU:
                    loss = loss.cuda()

                correct = label_vars.eq(torch.round(predictions.view(-1)))
                accuracy = correct.float().mean().data[0]
                avg_loss += loss.data[0]
                avg_accuracy += accuracy

                if (i+1) % 200 == 0:
                    print('Batch : ', i + 1, '/', total_batch,
                          ', MSE in this minibatch: ', round(loss.data[0], 3),
                          ', Accuracy:', round(accuracy, 2), ', time: %.2f' % (time.time() - t0))
                    t0 = time.time()
                if (i+1) >= total_eval_batch:
                    break

            if total_eval_batch > 0:
                eval_accuracy = float(avg_accuracy / total_eval_batch)
                eval_loss = float(avg_loss/total_eval_batch)
                print('\t  eval_loss: %.3f' % eval_loss, 'accuracy: %.2f' % eval_accuracy,
                        ', time: %.1fs' % (time.time() - t1))
            else:
                eval_accuracy = 0
                eval_loss = 0

            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=train_loss, train_accuracy=train_accuracy,
                        eval__loss=eval_loss, eval_accuracy=eval_accuracy,
                        step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()[:config.max_dataset]
        res = nsml.infer(reviews)
        print(res)

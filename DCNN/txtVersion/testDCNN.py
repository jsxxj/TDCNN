# -*- coding:utf-8 -*-
from __future__ import division
import os
import sys
import cPickle

import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

'''载入训练好的参数'''
def load_params(params_file):
    f = open(params_file,'rb')

    layer0_params = cPickle.load(f)
    layer1_params = cPickle.load(f)
    layer2_params = cPickle.load(f)
    layer3_params = cPickle.load(f)

    f.close()

    return layer0_params,layer1_params,layer2_params,layer3_params

def load_data(dataPath):
    data = numpy.loadtxt(dataPath)
    data_asarray = numpy.asarray(data,dtype='float64')
    data_x = data_asarray[:,:-1]
    data_y = data_asarray[:,-1]
    return data_x,data_y

class LeNetConvPoolLayer(object):
    def __init__(self,input,params_W,params_b,filter_shape,image_shape,poolsize=(2,2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = params_W
        self.b = params_b
        conv_out = conv.conv2d(input,
                               filters=self.W,
                               filter_shape=filter_shape,
                               image_shape= image_shape)
        pooled_out = downsample.max_pool_2d(
            input = conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out+self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W,self.b]

class HiddenLayer(object):
    def __init__(self,input, params_W,params_b,n_in,n_out,activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_out = T.dot(input,self.W) + self.b
        self.output = (
            lin_out if activation is None
            else activation(lin_out)
        )

        self.params = [self.W,self.b]

class LogisticRegression(object):
    def __init__(self,input,params_W,params_b,n_in,n_out):
        self.W = params_W
        self.b = params_b
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
        self.params = [self.W,self.b]

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def test_DCNN(dataset='testData.txt',params_file='params.pkl',nkerns=[5,10]):
    data,label = load_data(dataset)
    print label
    data_num = data.shape[0]

    layer0_params,layer1_params,layer2_params,layer3_params = load_params(params_file)
    x = T.matrix('x')

    layer0_input = x.reshape((data_num,1,40,36))
    layer0 = LeNetConvPoolLayer(
        input =  layer0_input,
        params_W = layer0_params[0],
        params_b = layer0_params[1],
        image_shape=(data_num,1,40,36),
        filter_shape=(nkerns[0],1,5,5),
        poolsize=(2,2)
    )
    layer1 = LeNetConvPoolLayer(
        input = layer0.output,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        image_shape=(data_num,nkerns[0],18,16),
        filter_shape=(nkerns[1],nkerns[0],5,5),
        poolsize=(2,2)
    )
    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        input = layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        n_in=nkerns[1]*7*6,
        n_out=2000,
        activation=T.tanh
    )

    layer3 = LogisticRegression(
        input = layer2.output,
        params_W=layer3_params[0],
        params_b=layer3_params[1],
        n_in = 2000,
        n_out = 3
    )

    f = theano.function(
        [x],
        layer3.y_pred,
        allow_input_downcast=True
    )
    pred = f(data)

    print 'Program complete.'

    print 'the Predicting the wrong sample output'
    count = 0
    for i in range(data_num):
        if pred[i] != label[i]:
            count = count + 1
            #print('data: %i is label %i, but mis-predicted as label %i' % (i, label[i], pred[i]))

    print 'Program accuary:'
    
    acc = count / data_num

    print 'the accuracy is %f%%' % ((1 - acc) * 100)

# -*- coding: utf-8 -*-

from __future__ import division
import os 
import sys
from theano.tensor.nnet import conv
import theano.tensor as T
import numpy, theano
import time,cPickle, gzip
from theano.tensor.signal import downsample
import LogisticRegression
from LogisticRegression import *
import sys
import matplotlib.pyplot as plt;plt.rcdefaults()  
import numpy as np

def load_params(params_files):
    f =open(params_files,'rb')
    layer0_params = cPickle.load(f)
    layer1_params = cPickle.load(f)
    layer2_params = cPickle.load(f)
    layer3_params = cPickle.load(f)
    return layer0_params,layer1_params,layer2_params,layer3_params

def load_data(dataset):

    print('... loading data ...')

    # Load the dataset
    with open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


class LeNetConvPoolLayer(object):
    def __init__(self,input,params_W,params_b, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = params_W
        self.b = params_b
        # 卷积
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        # 子采样
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

class HiddenLayer(object):
    def __init__(self, input, params_W,params_b, n_in, n_out,
                 activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

class LogisticRegression(object):
    def __init__(self, input, params_W,params_b,n_in, n_out):
        self.W = params_W
        self.b = params_b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

        

def test_CNN(dataset='txtData2_2.pkl',params_file='CNN_params.pkl'):

    dataset='txtData2_2.pkl'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    layer0_params,layer1_params,layer2_params,layer3_params = load_params(params_file)   
    x = T.matrix()
    f =open('Label.txt','r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split(',')
    f.close()
    
    print '... testing the model ...'
    
    # transfrom x from (batchsize, 28*28) to (batchsize,feature,28,28))
    # I_shape = (28,28),F_shape = (5,5),
    #第一层卷积、池化后  第一层卷积核为20个，每一个样本图片都产生20个特征图，
    N_filters_0 = 20
    D_features_0= 1
    #输入必须是为四维的，所以需要用到reshape，这一层的输入是一批样本是20个样本，28*28，
    
    layer0_input = x.reshape((150,1,40,36))
    layer0 = LeNetConvPoolLayer(
                                input = layer0_input, 
                                params_W = layer0_params[0],
                                params_b = layer0_params[1],
                                image_shape = (150,1,40,36),
                                filter_shape = (N_filters_0,1,5,5),
                                poolsize=(2, 2)
    )
    #layer0.output: (batch_size, N_filters_0, (28-5+1)/2, (28-5+1)/2) -> 20*20*12*12
    #卷积之后得到24*24 在经过池化以后得到12*12. 最后输出的格式为20个样本，20个12*12的特征图。卷积操作是对应的窗口呈上一个卷积核参数 相加在求和得到一个特
    #征图中的像素点数  这里池化采用最大池化 减少了参数的训练。
    N_filters_1 = 50
    D_features_1 = N_filters_0
    layer1 = LeNetConvPoolLayer(
                                input = layer0.output, 
                                params_W = layer1_params[0],
                                params_b = layer1_params[1],
                                image_shape = (150,N_filters_0,18,16),
                                filter_shape = (N_filters_1,D_features_1,5,5),
                                poolsize=(2, 2)                         
                                )
    # layer1.output: (20,50,4,4)
    #第二层输出为20个样本，每一个样本图片对应着50张4*4的特征图，其中的卷积和池化操作都是同第一层layer0是一样的。
    #这一层是将上一层的输出的样本的特征图进行一个平面化，也就是拉成一个一维向量，最后变成一个20*800的矩阵，每一行代表一个样本，
    
    #(20,50,4,4)->(20,(50*4*4))
    layer2_input = layer1.output.flatten(2) 
    #上一层的输出变成了20*800的矩阵，通过全连接，隐层操作，将800变成了500个神经元，里面涉及到全连接。
    layer2 = HiddenLayer(
                         layer2_input,
                         params_W=layer2_params[0],
                         params_b=layer2_params[1],
                         n_in = 50*7*5,
                         n_out = 500, 
                         activation = T.tanh
                         )
    
    #这里为逻辑回归层，主要是softmax函数作为输出，
    layer3 = LogisticRegression(input = layer2.output, 
                                params_W=layer3_params[0],
                                params_b=layer3_params[1],
                                n_in = 500, 
                                n_out = 3)
   
    ##########################
    #预测函数
    f = theano.function(
        [x],    #funtion 的输入必须是list，即使只有一个输入
        layer3.y_pred,
        allow_input_downcast=True
    )
    
    #预测的类别pred
    pred = f(test_set_x[:])
    #将预测的类别pred与真正类别label对比，输出错分的图像
    count = 0
    list =[]
    for i in range(len(test_set_x)): 
        if float(pred[i]) == float(line[i]):
            count = count +1
            list.append(str(pred[i]))
            #print('data: %i is label %i, but mis-predicted as label %i' %(i, line[i], pred[i]))
    Label = set(line)
    predValue = set(list)
    Lable_num = []
    predValue_num = []
    print '... 训练集标签 ...'
    print sorted(Label)
    for i in sorted(Label):
        Lable_num.append(line.count(i))
        
    print '... 训练集标签对应的类别总数 ...'
    print Lable_num
    
    
    for j in sorted(predValue):
        predValue_num.append(list.count(j))
    
    print '... 模型预测正确的类别总数 ...'
    print predValue_num
    
    class_rate = []
    for i in range(10):
        class_rate.append(str(round(predValue_num[i]/Lable_num[i],4)))
    fw = open('CNN_acc.txt','wb')
    for i in range(10):
        fw.write(str(class_rate[i]))
        fw.write(" ")
    fw.close()
    print '... 模型预测各类别的正确率 ...'
    print class_rate
    print('... Program complete ...')
    print ('测试集数据样本总数为:' )
    print (10000)
    print ('识别正确的数据样本数为:' )
    print (count)
    acc = count/10000  
    print("the accuracy rate is : %f %%" % (acc*100)) 
    
    method = ('0','1','2','3','4','5','6','7','8','9')  
    x_pos = np.arange(10)    
    plt.bar(x_pos,class_rate,alpha = 0.4)  
    plt.xticks(x_pos,method)  
    plt.title('CNN Handwriting Recognition Accuracy Rate Histogram')  
    plt.show()     

if __name__ == '__main__':
    test_CNN()
 
        

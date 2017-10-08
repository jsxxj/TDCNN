# -*- coding: utf-8 -*-
from theano.tensor.nnet import conv
import theano.tensor as T
import numpy, theano
import time,cPickle, gzip
import os 
import sys 
from theano.tensor.signal import downsample
from mlp import HiddenLayer
import LogisticRegression
from LogisticRegression import *
from numpy import save
import cPickle 


class LeNetConvPoolLayer (object):
    def __init__ (self, rng, input, filter_shape, image_shape, poolsize=(2,2)):
        assert filter_shape[1]==image_shape[1]
        self.input = input #(batches, feature, Ih, Iw)
        
        fan_in = numpy.prod(filter_shape[1:])#number of connections for each filter
        W_value = numpy.asarray(rng.uniform(low = -numpy.sqrt(3./fan_in), high = numpy.sqrt(3./fan_in),size = filter_shape),
                                dtype = theano.config.floatX)
        self.W = theano.shared(W_value,name = 'W') #(filters, feature, Fh, Fw)
        
        b_value = numpy.zeros((filter_shape[0],),dtype = theano.config.floatX)
        self.b = theano.shared(b_value, name = 'b')
        
        conv_res = conv.conv2d(input,self.W,image_shape = image_shape, filter_shape = filter_shape) #(batches, filters, Ih-Fh+1, Iw-Fw+1)
        pooled = downsample.max_pool_2d(conv_res,poolsize)
        self.output = T.tanh(pooled + self.b.dimshuffle('x',0,'x','x'))
        
        self.params = [self.W, self.b]
        
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):      
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.L1_1 = abs((self.W).sum())
        self.L2_1 = (self.W**2).sum()
        # parameters of the model
        self.params = [self.W, self.b]     

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.L1_2 = abs(self.W).sum()
        self.L2_2 = (self.W**2).sum()
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):        
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])       

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
def load_data(dataset):

    print('... loading data')
    
    f = open(dataset,'rb')
    train_set = cPickle.load(f)
    valid_set = cPickle.load(f)
    test_set = cPickle.load(f)

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
            
def save_params(param1,param2,param3,param4):  
        write_file = open('CNN_params.pkl', 'wb')
        cPickle.dump(param1, write_file, -1)
        cPickle.dump(param2, write_file, -1)
        cPickle.dump(param3, write_file, -1)
        cPickle.dump(param4, write_file, -1)
        write_file.close()         

def test_CNN(learning_rate = 0.01, L1_reg=0.00, L2_reg=0.0001,n_epochs = 1000, batch_size = 20, n_hidden = 500,dataset='txtData7.pkl'):
    dataset = load_data(dataset)
    ''' train_set_x.get_value(); tt.shape ---(50000, 784)'''
    train_set_x, train_set_y = dataset[0] 
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    print ('training set has %i batches' %n_train_batches)
    print ('validate set has %i batches' %n_valid_batches)
    print ('testing set has %i batches' %n_test_batches)
    
    #symbolic variables
    x = T.matrix()
    y = T.ivector() #lvector: [long int] labels; ivector:[int] labels
    minibatch_index = T.lscalar()
    
    print 'build the model...'
    rng = numpy.random.RandomState(23455)

    # transfrom x from (batchsize, 28*28) to (batchsize,feature,28,28))
    # I_shape = (28,28),F_shape = (5,5),
    #第一层卷积、池化后  第一层卷积核为20个，每一个样本图片都产生20个特征图，
    N_filters_0 = 20
    D_features_0= 1
    #输入必须是为四维的，所以需要用到reshape，这一层的输入是一批样本是20个样本，28*28，
    layer0_input = x.reshape((batch_size,D_features_0,40,36)) 
    layer0 = LeNetConvPoolLayer(rng, 
                                input = layer0_input, 
                                filter_shape = (N_filters_0,D_features_0,5,5),
                                image_shape = (batch_size,1,40,36))
    #layer0.output: (batch_size, N_filters_0, (28-5+1)/2, (28-5+1)/2) -> 20*20*12*12
    #卷积之后得到24*24 在经过池化以后得到12*12. 最后输出的格式为20个样本，20个12*12的特征图。卷积操作是对应的窗口呈上一个卷积核参数 相加在求和得到一个特
    #征图中的像素点数  这里池化采用最大池化 减少了参数的训练。
    N_filters_1 = 50
    D_features_1 = N_filters_0
    layer1 = LeNetConvPoolLayer(rng,
                                input = layer0.output, 
                                filter_shape = (N_filters_1,D_features_1,5,5),
                                image_shape = (batch_size,N_filters_0,18,16))
    # layer1.output: (20,50,4,4)
    #第二层输出为20个样本，每一个样本图片对应着50张4*4的特征图，其中的卷积和池化操作都是同第一层layer0是一样的。
    #这一层是将上一层的输出的样本的特征图进行一个平面化，也就是拉成一个一维向量，最后变成一个20*800的矩阵，每一行代表一个样本，
    
    #(20,50,4,4)->(20,(50*4*4))
    layer2_input = layer1.output.flatten(2) 
    #上一层的输出变成了20*800的矩阵，通过全连接，隐层操作，将800变成了500个神经元，里面涉及到全连接。
    layer2 = HiddenLayer(rng,
                         layer2_input,
                         n_in = 50*7*6,
                         n_out = 500, 
                         activation = T.tanh)
    #这里为逻辑回归层，主要是softmax函数作为输出，
    layer3 = LogisticRegression(input = layer2.output, 
                                n_in = 500, 
                                n_out = 3)
    #约束规则
   
    ##########################
    cost = (layer3.negative_log_likelihood(y)
            +L1_reg * (layer2.L1_1 + layer3.L1_2)
            +L2_reg * (layer2.L2_1 + layer3.L2_2))

    test_model = theano.function(inputs = [minibatch_index],
                                 outputs = layer3.errors(y),
                                 givens = {
                                     x: test_set_x[minibatch_index*batch_size : (minibatch_index+1) * batch_size],
                                     y: test_set_y[minibatch_index*batch_size : (minibatch_index+1) * batch_size]})
    
    valid_model = theano.function(inputs = [minibatch_index],
                  outputs = layer3.errors(y),
                  givens = {
                    x: valid_set_x[minibatch_index * batch_size : (minibatch_index+1) * batch_size],
                    y: valid_set_y[minibatch_index * batch_size : (minibatch_index+1) * batch_size]})
    
    params = layer3.params + layer2.params + layer1.params + layer0.params
    gparams = T.grad(cost,params)
    
    updates = []
    for par,gpar in zip(params,gparams):
        updates.append((par, par - learning_rate * gpar))
    
    train_model = theano.function(inputs = [minibatch_index],
                                  outputs = [cost],
                                  updates = updates,
                                  givens = {x: train_set_x[minibatch_index * batch_size : (minibatch_index+1) * batch_size],
                                            y: train_set_y[minibatch_index * batch_size : (minibatch_index+1) * batch_size]})
    
    
    #---------------------Train-----------------------#
    print 'training...'
    
    print ('training set has %i batches' %n_train_batches)
    print ('validate set has %i batches' %n_valid_batches)
    print ('testing set has %i batches' %n_test_batches)

    epoch = 0
    patience = 10000
    patience_increase = 2
    validation_frequency = min(n_train_batches,patience/2)
    improvement_threshold = 0.995
    
    best_parameters = None
    min_validation_error = numpy.inf
    done_looping = False
    
    start_time = time.clock()
    while (epoch<n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
        
            #cur_batch_train_error,cur_params = train_model(minibatch_index)
            cur_batch_train_error = train_model(minibatch_index)
            iter = (epoch-1) * n_train_batches + minibatch_index
            
            if (iter+1)%validation_frequency ==0:
                #validation_error = numpy.mean([valid_model(idx) for idx in xrange(n_valid_batches)])
                validation_losses = [valid_model(i) for i
                                     in xrange(n_valid_batches)]
                validation_error = numpy.mean(validation_losses)
                
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      validation_error * 100.))
                
                if validation_error < min_validation_error:
                    if validation_error < min_validation_error * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    min_validation_error = validation_error
                    
                    #best_parameters = cur_params
                    best_iter = iter
                    
                    save_params(layer0.params,layer1.params,layer2.params,layer3.params)
                    
                    test_error = numpy.mean([test_model(idx) for idx in xrange(n_test_batches)])
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_error * 100.))
            
            
            if iter>=patience:
                done_looping = True
                break
            
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          ( 100-min_validation_error* 100., best_iter + 1, 100-test_error * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    test_CNN()

 
        

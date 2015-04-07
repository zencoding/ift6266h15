import os
import sys
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layer import ReluConv2DLayer, MaxPoolingLayer, ReluLayer, LinearLayer
from classifier import LogisticRegression
from train import GraddescentMinibatch, Dropout
from params import save_params, load_params, set_params, get_params

import pdb


#######################
# SET SUPER PARAMETER #
#######################

batchsize = 100
momentum = 0.9

weightdecay = 0.01
finetune_lr = 1e-2
finetune_epc = 400

print " "
print "batchsize =", batchsize
print "momentum =", momentum
print "finetune:            lr = %f, epc = %d" % (finetune_lr, finetune_epc)

#############
# LOAD DATA #
#############

print "... preparing data"
class data_generator(object):
    def __init__(self, string, truthstring, numpart=8):
        self.string = string
        self.truthstring = truthstring
        self.partidx = 0
        self.numpart = numpart
    
    def __iter__(self):
        return self

    def next(self):
        if self.partidx < self.numpart:
            filename = self.string + '%d.npy' % self.partidx
            truthname = self.truthstring + '%d.npy' % self.partidx
            data = numpy.load(filename)
            truth = numpy.load(truthname)
            self.partidx += 1
            return (data, truth)
        else:
            self.partidx = 0
            raise StopIteration
    
    def reset(self):
        self.partidx = 0


train_dg = data_generator('train_padnormbc01_part', 'train_truth_part', 8)
test_dg = data_generator('valid_padnormbc01_part', 'valid_truth_part', 2)

x, y = train_dg.next()
train_dg.reset()
train_x = theano.shared(value=x, name='train_x', borrow=True)
train_y = theano.shared(value=y, name='train_y', borrow=True)
x, y = test_dg.next()
test_dg.reset()
test_x = theano.shared(value=x, name='test_x', borrow=True)
test_y = theano.shared(value=y, name='test_y', borrow=True)
print "Done."

###############
# BUILD MODEL #
###############

print "... building model"
l0_n_in = (batchsize, 3, 250, 250)
l0_filter_shape=(32, l0_n_in[1], 5, 5)
l0_pool_n_in = (batchsize, l0_filter_shape[0], l0_n_in[2]-l0_filter_shape[2]+1, l0_n_in[3]-l0_filter_shape[3]+1)
l0_poolsize = (3, 3)

l1_n_in = (batchsize, l0_filter_shape[0], l0_pool_n_in[2]/l0_poolsize[0], l0_pool_n_in[3]/l0_poolsize[1])
l1_filter_shape=(48, l1_n_in[1], 5, 5)
l1_pool_n_in = (batchsize, l1_filter_shape[0], l1_n_in[2]-l1_filter_shape[2]+1, l1_n_in[3]-l1_filter_shape[3]+1)
l1_poolsize = (3, 3)

l2_n_in = (batchsize, l1_filter_shape[0], l1_pool_n_in[2]/l1_poolsize[0], l1_pool_n_in[3]/l1_poolsize[1])
l2_filter_shape=(64, l2_n_in[1], 3, 3)
l2_pool_n_in = (batchsize, l2_filter_shape[0], l2_n_in[2]-l2_filter_shape[2]+1, l2_n_in[3]-l2_filter_shape[3]+1)
l2_poolsize = (3, 3)

l3_n_in = (batchsize, l2_filter_shape[0], l2_pool_n_in[2]/l2_poolsize[0], l2_pool_n_in[3]/l2_poolsize[1])
l3_filter_shape=(128, l3_n_in[1], 3, 3)
l3_pool_n_in = (batchsize, l3_filter_shape[0], l3_n_in[2]-l3_filter_shape[2]+1, l3_n_in[3]-l3_filter_shape[3]+1)
l3_poolsize = (3, 3)

npy_rng = numpy.random.RandomState(123)
model = ReluConv2DLayer(
    n_in=l0_n_in, filter_shape=l0_filter_shape, npy_rng=npy_rng
) + MaxPoolingLayer(
    n_in=l0_pool_n_in, pool_size=l0_poolsize, ignore_border=True
) + ReluConv2DLayer(
    n_in=l1_n_in, filter_shape=l1_filter_shape, npy_rng=npy_rng
) + MaxPoolingLayer(
    n_in=l1_pool_n_in, pool_size=l1_poolsize, ignore_border=True
) + ReluConv2DLayer(
    n_in=l2_n_in, filter_shape=l2_filter_shape, npy_rng=npy_rng
) + MaxPoolingLayer(
    n_in=l2_pool_n_in, pool_size=l2_poolsize, ignore_border=True
) + ReluConv2DLayer(
    n_in=l3_n_in, filter_shape=l3_filter_shape, npy_rng=npy_rng
) + MaxPoolingLayer(
    n_in=l3_pool_n_in, pool_size=l3_poolsize, ignore_border=True
) + ReluLayer(512, 512, npy_rng=npy_rng
) + ReluLayer(512, 256, npy_rng=npy_rng
) + ReluLayer(256, 128, npy_rng=npy_rng
) + LogisticRegression(128, 10, npy_rng=npy_rng)

model.print_layer()
# load_params(model, 'CONV_apr_4.npy')

# compile error rate counters:
index = T.lscalar()
truth = T.lvector('truth')
train_set_error_rate = theano.function(
    [index],
    T.mean(T.neq(model.models_stack[-1].predict(), truth)),
    givens = {model.varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
)
def train_error():
    total_mean = 0.
    mem = train_dg.partidx
    train_dg.reset()
    for ipart in train_dg:
        train_x.set_value(ipart[0])
        train_y.set_value(ipart[1])
        total_mean += numpy.mean([train_set_error_rate(i) for i in xrange(25)])
    train_dg.partidx = mem
    return total_mean / 8.

test_set_error_rate = theano.function(
    [index],
    T.mean(T.neq(model.models_stack[-1].predict(), truth)),
    givens = {model.varin : test_x[index * batchsize: (index + 1) * batchsize],
              truth : test_y[index * batchsize: (index + 1) * batchsize]},
)
def test_error():
    total_mean = 0.
    mem = test_dg.partidx
    test_dg.reset()
    for ipart in test_dg:
        test_x.set_value(ipart[0])
        test_y.set_value(ipart[1])
        total_mean += numpy.mean([test_set_error_rate(i) for i in xrange(25)])
    test_dg.partidx = mem
    return total_mean / 2.
print "Done."
print "Initial error rate: train: %f, test: %f" % (train_error(), test_error())

#############
# FINE-TUNE #
#############

print "\n\n... fine-tuning the whole network"
trainer = GraddescentMinibatch(
    varin=model.varin, data=train_x, 
    truth=model.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=model.models_stack[-1].cost(), 
    params=model.params,
    batchsize=batchsize, learningrate=finetune_lr, momentum=momentum,
    rng=npy_rng
)

init_lr = trainer.learningrate
prev_cost = numpy.inf
for epoch in xrange(finetune_epc):
    i = 0
    cost = 0.
    for ipart in train_dg:
        print "part %d " % i,
        train_x.set_value(ipart[0])
        train_y.set_value(ipart[1])
        i += 1
        cost += trainer.step()
        
        # horizontal flip
        train_x.set_value(ipart[0][:, :, :, ::-1])
        print "       ",
        cost += trainer.step()
        
        # vertical flip
        train_x.set_value(ipart[0][:, :, ::-1, :])
        print "       ",
        cost += trainer.step()

        # 180 rotate
        train_x.set_value(ipart[0][:, :, ::-1, ::-1])
        print "       ",
        cost += trainer.step()

        # right rotate
        rotate = numpy.swapaxes(ipart[0], 2, 3)
        train_x.set_value(rotate)
        print "       ",
        cost += trainer.step()

        # right rotate filp
        train_x.set_value(rotate[:, :, ::-1, :])
        print "       ",
        cost += trainer.step()

        # left rotate
        train_x.set_value(rotate[:, :, ::-1, ::-1])
        print "       ",
        cost += trainer.step()
        
        # left rotate filp
        train_x.set_value(rotate[:, :, :, ::-1])
        print "       ",
        cost += trainer.step()

    cost /= i * 8.
    if prev_cost <= cost:
        if trainer.learningrate < (init_lr * 1e-7):
            break
        trainer.set_learningrate(trainer.learningrate*0.8)
    prev_cost = cost
    print "*** epoch %d cost: %f" % (epoch, cost)
    print "*** error rate: train: %f, test: %f" % (train_error(), test_error())
    try:
        if epoch % 30 == 0:
            save_params(model, 'CONV_5-5-3-3_32-48-64-128_3333_512-512-256-128-2_dtagmt.npy')
    except:
        pass
print "***FINAL error rate: train: %f, test: %f" % (train_error(), test_error())
print "Done."

pdb.set_trace()

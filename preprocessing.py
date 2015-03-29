import numpy
import theano
import theano.tensor as T
from skimage.transform import downscale_local_mean as downscale

from layer import ReluConv2DLayer, MaxPoolingLayer, SigmoidLayer
from preprocess import ZCA
from dataset import CatsnDogs

import pdb


def Padding(images, padsize=(500, 500, 3), pool=2):
    """
    I hate this form of implementation, improve it to be more terse.

    Accepts a part yielded by the dataset wrapper, and returns a theano shared
    variable of data.
    
    Parameters:
    -------------
    pool : int
    pool must be a dividable number of padsize. It downsamples the padded
    image with pool as factor. 

    """
    outputsize = (len(images), padsize[0]/pool, padsize[1]/pool, 3)
    output = numpy.zeros(outputsize, dtype=theano.config.floatX)
    template = numpy.zeros(padsize, dtype=theano.config.floatX)
    
    for i in xrange(len(images)):
        img = images[i]
        height, width, _ = img.shape
        
        hdiff = height - padsize[0]
        hm = hdiff / 2  # -(hdiff/2) != -hdiff/2 while hdiff is odd.
        wdiff = width - padsize[1]
        wm = wdiff / 2
        if hdiff >= 0 and wdiff >= 0:  # image completely larger
            template[:, :, :] = img[
                hm : hm + padsize[0],
                wm : wm + padsize[1], :]
        
        elif hdiff >= 0 and wdiff < 0: # taller but narrower
            template[:, -wm : width - wm, :] = img[
                hm : hm + padsize[0], :, :]
            
            wleftmargin = -wm
            wrightmargin = width - wm
            while wleftmargin > width:
                template[:, wleftmargin - width : wleftmargin, :] = img[
                    hm : hm + padsize[0], :, :]
                template[:, wrightmargin : wrightmargin + width, :] = img[
                    hm : hm + padsize[0], :, :]
                wleftmargin -= width
                wrightmargin += width
            template[:, : wleftmargin, :] = img[
                hm : hm + padsize[0], width - wleftmargin:, :]
            template[:, wrightmargin:, :] = img[
                hm : hm + padsize[0], :padsize[1] - wrightmargin, :]
            
        elif hdiff < 0 and wdiff >= 0: # shorter but wider
            template[-hm : height - hm, :, :] = img[:, wm : wm + padsize[0], :]
            
            hleftmargin = -hm
            hrightmargin = height - hm
            while hleftmargin > height:
                template[hleftmargin - height : hleftmargin, :, :] = img[
                    :, wm : wm + padsize[0], :]
                template[hrightmargin : hrightmargin + height, :, :] = img[
                    :, wm : wm + padsize[0], :]
                hleftmargin -= height
                hrightmargin += height
            template[:hleftmargin, :, :] = img[
                height - hleftmargin:, wm : wm + padsize[0], :]
            template[hrightmargin:, :, :] = img[
                :padsize[0] - hrightmargin, wm : wm + padsize[0], :]

        else:  # completely smaller
            template[-hm : height - hm,
                     -wm : width - wm, :] = img[:, :, :]

            wleftmargin = -wm
            wrightmargin = width - wm
            while wleftmargin > width:
                template[-hm : height - hm,
                         wleftmargin - width : wleftmargin, :] = img[:, :, :]
                template[-hm : height - hm,
                         wrightmargin : wrightmargin + width, :] = img[:, :, :]
                wleftmargin -= width
                wrightmargin += width
            template[-hm : height - hm, : wleftmargin, :] = img[
                :, width - wleftmargin:, :]
            template[-hm : height - hm, wrightmargin:, :] = img[
                :, :padsize[1] - wrightmargin, :]
 
            hleftmargin = -hm
            hrightmargin = height - hm
            while hleftmargin > height:
                template[hleftmargin - height : hleftmargin, :, :] = template[
                    -hm : height - hm, :, :]
                template[hrightmargin : hrightmargin + height, :] = template[
                    -hm : height - hm, :, :]
                hleftmargin -= height
                hrightmargin += height
            template[:hleftmargin, :, :] = template[
                -hm + height - hleftmargin : -hm + height, :, :]
            template[hrightmargin:, :, :] = template[
                -hm : -hm + padsize[0] - hrightmargin, :, :]

        output[i] = downscale(template, (pool, pool, 1))
    output = theano.shared(value=output, name='padedpart', borrow=True)
    return output
    """
        A terse but maybe slow implementation:
        hrepeat = (2 * padsize[0] - 1) / height + 1
        wrepeat = (2 * padsize[1] - 1) / width + 1
        large = numpy.tile(img, (hrepeat, wrepeat, 1))
        
        hmargin = - (padsize[0] - height) / 2
        wmargin = - (padsize[1] - width) / 2
        while hmargin < 0:
            hmargin += height
        while wmargin < 0:
            wmargin += width
        output[i] = large[hmargin:hmargin+padsize[0],
                          wmargin:wmargin+padsize[1], :]
    output = theano.shared(value=output, name='padedpart', borrow=True)
    return output
    """




#############
# LOAD DATA #
#############

npy_rng = numpy.random.RandomState(123)
data_wrapper = CatsnDogs(partsize=100, npy_rng=npy_rng)

train_parts = data_wrapper.train_generator()
valid_parts = data_wrapper.valid_generator()

try:
    ipart = train_parts.next()
    part_data = ipart[0]
    part_truth = ipart[1]

    #################
    # PAD/POOL DATA #
    #################
    
    paded_part = Padding(part_data)
    
    #######
    # ZCA #
    #######

    pdb.set_trace()

    ##############
    # STORE FLAG #
    ##############

except StopIteration:
    pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:33:09 2022

@author: mirfan
"""

from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization, UpSampling2D
from keras.models import Model, Sequential
from keras.layers import LeakyReLU

def generate_model(image_shape):
    
    kernel_size = 8
    channels = 1
    mom = 0.5
    alp = 0.2
    
    model = Sequential() 
    model.add(Conv2D(8, kernel_size=kernel_size, padding="same")) 
    model.add(LeakyReLU(alpha=alp)) 
    model.add(BatchNormalization(momentum=mom))
    model.add(Conv2D(16, kernel_size=kernel_size, padding="same", strides=2)) 
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(Conv2D(32, kernel_size=kernel_size, padding="same", strides=2)) 
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(Conv2D(64, kernel_size=kernel_size, padding="same", strides=2)) 
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(Conv2D(128, kernel_size=kernel_size, padding="same", strides=2)) 
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(Conv2D(256, kernel_size=kernel_size, padding="same", strides=2))
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(UpSampling2D())
    model.add(Conv2D(16, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(UpSampling2D())
    model.add(Conv2D(8, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=alp))
    model.add(BatchNormalization(momentum=mom))
    model.add(Conv2D(channels, kernel_size=kernel_size, padding="same"))
    model.add(Activation("tanh"))
    img_in = Input(shape=image_shape)
    img_out = model(img_in)
    return Model(img_in, img_out)
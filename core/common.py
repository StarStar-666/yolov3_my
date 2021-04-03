#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:12:53
#   Description :
#
#================================================================

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)


    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def separable_conv(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.DepthwiseConv2D( kernel_size=filters_shape[0], strides=strides,padding=padding,use_bias = not bn, depthwise_initializer = tf.random_normal_initializer(stddev=0.01),
                                               bias_initializer = tf.constant_initializer(0.), depthwise_regularizer = tf.keras.regularizers.l2(0.0005))(input_layer)
    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = 1, strides=1, padding='same',
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(conv)


    if bn: conv = BatchNormalization()(conv)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

def conv2d_bn(x,filters,num_row,num_col,padding='same',stride=1,dilation_rate=1,relu=True):
    x =  tf.keras.layers.Conv2D(
        filters, (num_row, num_col),
        strides=(stride,stride),
        padding=padding,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization()(x)
    if relu:
        x = tf.nn.leaky_relu(x, alpha=0.1)
    return x

def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')



def RFB_net(input_data ,name, stride=1,scale =0.1, activate=True, bn=True):


    conv =  convolutional(input_data, (1, 1, 1024, 256), downsample=False, activate=True, bn=True)

    conv_1 = separable_conv(conv, (3,3,256,256))

    conv_1_1 = conv2d_bn(conv_1 ,256 ,3 ,3 ,dilation_rate=3)

    conv_2_1 = conv2d_bn(conv_1,256, 3, 3, dilation_rate=5)

    output = tf.concat([conv_1, conv_1_1, conv_2_1, conv], axis=3)

    output = convolutional(output, (1, 1, 1024, 1024), downsample=False, activate=False, bn=True)

    output = output + input_data

    output  = tf.nn.leaky_relu(output, alpha=0.1)

    return output

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras.layers import Lambda


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    if K.image_data_format() == "channels_first":

        channel_axis = 1
    else:
        channel_axis = -1
    channel = input_feature.get_shape()[-1]

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.get_shape()[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.get_shape()[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.get_shape()[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.get_shape()[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.get_shape()[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.get_shape()[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.get_shape()[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.get_shape()[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.get_shape()[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])
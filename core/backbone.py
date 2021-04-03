#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:37:51
#   Description :
#
# ================================================================

import tensorflow as tf
import core.common as common


def darknet53(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 32))
    input_data = common.convolutional(input_data, (3, 3, 32, 64), downsample=True)

    for i in range(1):
        input_data = common.residual_block(input_data, 64, 32, 64)

    input_data = common.convolutional(input_data, (3, 3, 64, 128), downsample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 128, 64, 128)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def _stem_block(input_x):
    conv0 = common.convolutional(input_x, (3, 3, 3, 32), downsample=True, activate=True, bn=True)

    conv1_l0 = common.convolutional(conv0, (1, 1, 32, 16), downsample=False, activate=True, bn=True)
    conv1_l1 = common.convolutional(conv1_l0, (3, 3, 16, 32), downsample=True, activate=True, bn=True)

    maxpool1_r0 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv0)

    filter_concat = tf.concat([conv1_l1, maxpool1_r0], axis=-1)

    output = common.convolutional(filter_concat, (1, 1, 64, 32), downsample=False, activate=True, bn=True)

    return output


def _dense_block(input_x, stage, num_block, k, bottleneck_width):
    output = input_x
    for index in range(num_block):
        inter_channel = k * bottleneck_width
        # left channel
        conv_left_0 = common.convolutional(output, (1, 1, tf.cast(tf.shape(input_x)[-1], tf.int32), inter_channel),
                                           downsample=False, activate=True, bn=True)
        conv_left_1 = common.convolutional(conv_left_0, (3, 3, inter_channel, k), downsample=False, activate=True,
                                           bn=True)
        conv_left_1 = common.cbam_block(conv_left_1)

        # right channel
        conv_right_0 = common.convolutional(output, (1, 1, tf.cast(tf.shape(input_x)[-1], tf.int32), inter_channel),
                                            downsample=False, activate=True, bn=True)
        conv_right_1 = common.convolutional(conv_right_0, (3, 3, inter_channel, k), downsample=False, activate=True,
                                            bn=True)
        conv_right_2 = common.convolutional(conv_right_1, (3, 3, k, k), downsample=False, activate=True, bn=True)

        conv_right_2 = common.cbam_block(conv_right_2)

        output = tf.concat([output, conv_left_1, conv_right_2], axis=3)

    return output


def _transition_layer(input_x, stage, output_channel, is_avgpool=True):

    conv0 = common.convolutional(input_x, (1, 1, tf.cast(tf.shape(input_x)[-1], tf.int32), output_channel),
                                 downsample=False, activate=True, bn=True)

    if is_avgpool:

        conv0 = common.separable_conv(input_x, (3, 3, output_channel, output_channel), downsample=True)

    output = conv0

    output = common.cbam_block(output)

    return output


def peleenet(input_data):

    stem_block_output = _stem_block(input_data)

    dense_block_output = _dense_block(stem_block_output, 0, 3, 16, 2)

    transition_layer_output = _transition_layer(dense_block_output, 0, 128)

    dense_block_output1 = _dense_block(transition_layer_output, 1, 4, 16, 2)

    transition_layer_output1 = _transition_layer(dense_block_output1, 1, 256)

    dense_block_output2 = _dense_block(transition_layer_output1, 2, 8, 16, 4)

    transition_layer_output2 = _transition_layer(dense_block_output2, 2, 512)

    dense_block_output3 = _dense_block(transition_layer_output2, 3, 6, 16, 4)

    transition_layer_output3 = _transition_layer(dense_block_output3, 3, 1024, is_avgpool=False)

    return dense_block_output1, dense_block_output2, transition_layer_output3






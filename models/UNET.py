#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FCN8 with VGG16 feature extractor
"""

import numpy as np
import sys
import tensorflow as tf


class UNET(tf.keras.Model):

    def __init__(self, n_classes=21, floatType=32, input_shape=(256, 256, 3)):
        super(UNET, self).__init__()

        if floatType == 32:
            self.floatType = tf.float32
        elif floatType == 16:
            tf.keras.backend.set_floatx('float16')
            self.floatType = tf.float16
        else:
            raise Exception('floatType should be either 32 or 16')

        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self.x = self.inputs

        # encoder part
        n_filters_list = [64, 128, 256, 512]
        intermediate_outputs = []
        for n_filters in n_filters_list:
            self.x = tf.keras.layers.Conv2D(
                n_filters, (3, 3), activation='relu', padding='same')(self.x)
            self.x = tf.keras.layers.Conv2D(
                n_filters, (3, 3), activation='relu', padding='same')(self.x)
            intermediate_outputs.append(self.x)
            self.x = tf.keras.layers.MaxPool2D(
                pool_size=(2, 2), strides=(2, 2), padding='same')(self.x)
            self.x = tf.keras.layers.Dropout(0.3)(self.x)

        # bottleneck
        self.x = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu',
                                        padding='same')(self.x)
        self.x = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu',
                                        padding='same')(self.x)

        # decoder part
        for idx, n_filters in enumerate(reversed(n_filters_list)):
            self.x = tf.keras.layers.Conv2DTranspose(
                n_filters, (3, 3), strides=(2, 2),
                activation='relu', padding='same')(self.x)
            self.x = tf.keras.layers.concatenate([
                self.x, intermediate_outputs[len(intermediate_outputs)-idx-1]])
            self.x = tf.keras.layers.Dropout(0.3)(self.x)
            self.x = tf.keras.layers.Conv2D(
                n_filters, (3, 3), activation='relu', padding='same')(self.x)
            self.x = tf.keras.layers.Conv2D(
                n_filters, (3, 3), activation='relu', padding='same')(self.x)

        self.outputs = tf.keras.layers.Conv2D(
            n_classes, (3, 3), activation='softmax', padding='same')(self.x)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

    def call(self, x):
        return self.model(x)

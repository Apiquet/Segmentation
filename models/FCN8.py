#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FCN8 with VGG16 feature extractor
"""

import numpy as np
import sys
import tensorflow as tf


class FCN8(tf.keras.Model):

    def __init__(self, tracker_ssd_path, ssd_weights_path=None,
                 n_classes=21, floatType=32, input_shape=(300, 300, 3)):
        """
        Args:
            - (str) tracker_ssd_path: path to github/Apiquet/Tracking_SSD_ReID
            - (str) ssd_weights_path: got from Tracking_SSD_ReID/training.ipynb
            - (int) n_classes: number of target classes
            - (int) floatType: if wanted to se float32 or 16
        """
        super(FCN8, self).__init__()

        if floatType == 32:
            self.floatType = tf.float32
        elif floatType == 16:
            tf.keras.backend.set_floatx('float16')
            self.floatType = tf.float16
        else:
            raise Exception('floatType should be either 32 or 16')

        sys.path.append(tracker_ssd_path)
        from models.SSD300 import SSD300

        self.n_classes = n_classes
        SSD300_model = SSD300(21, floatType)
        confs, locs = SSD300_model(tf.zeros([32, 300, 300, 3], self.floatType))
        if ssd_weights_path is not None:
            SSD300_model.load_weights(ssd_weights_path)
        SSD_backbone = SSD300_model.getVGG16()

        from models.VGG16 import VGG16
        self.input_res = input_shape
        self.VGG16 = VGG16(input_shape=input_shape)
        self.VGG16_tilStage5 = self.VGG16.getUntilStage5()

        ssd_seq_idx = 0
        ssd_layer_idx = 0
        for i in range(len(self.VGG16_tilStage5.layers)):
            ssd_layer_idx = i
            if i >= 13:
                ssd_seq_idx = 1
                ssd_layer_idx -= 13
            self.VGG16_tilStage5.get_layer(index=i).set_weights(
                SSD_backbone.get_layer(index=ssd_seq_idx).get_layer(
                    index=ssd_layer_idx).get_weights())
            self.VGG16_tilStage5.get_layer(index=i).trainable = True
        del SSD_backbone
        del SSD300_model

        self.inputs = tf.keras.layers.Input(shape=input_shape)
        self.x = self.VGG16_tilStage5.get_layer(index=0)(self.inputs)
        for i in range(1, 10):
            self.x = self.VGG16_tilStage5.get_layer(index=i)(self.x)
        self.out_stage_3 = self.x

        for i in range(10, 14):
            self.x = self.VGG16_tilStage5.get_layer(index=i)(self.x)
        self.out_stage_4 = self.x

        for i in range(14, len(self.VGG16_tilStage5.layers)):
            self.x = self.VGG16_tilStage5.get_layer(index=i)(self.x)

        self.x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                           strides=(2, 2),
                                           padding='same')(self.x)

        self.x = tf.keras.layers.Conv2DTranspose(
            n_classes, kernel_size=(4, 4),
            strides=(2, 2), use_bias=False)(self.x)
        self.x = tf.keras.layers.Cropping2D(
            cropping=((2, 1), (1, 2)))(self.x)

        self.out_stage_4_resized = tf.keras.layers.Conv2D(
            n_classes, (1, 1), activation='relu',
            padding='same')(self.out_stage_4)

        self.x = tf.keras.layers.Add()([self.x, self.out_stage_4_resized])

        self.x = tf.keras.layers.Conv2DTranspose(
            n_classes, kernel_size=(4, 4), strides=(2, 2),
            use_bias=False)(self.x)
        self.x = tf.keras.layers.Cropping2D(cropping=(1, 1))(self.x)

        self.out_stage_3_resized = tf.keras.layers.Conv2D(
            n_classes, (1, 1), activation='relu',
            padding='same')(self.out_stage_3)

        self.x = tf.keras.layers.Add()([self.x, self.out_stage_3_resized])

        self.x = tf.keras.layers.Conv2DTranspose(
            n_classes, kernel_size=(8, 8), strides=(8, 8),
            use_bias=False)(self.x)
        self.x = tf.keras.layers.Cropping2D(cropping=(2, 2))(self.x)

        self.outputs = tf.keras.layers.Activation('softmax')(self.x)

        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

    def call(self, x):
        return self.model(x)

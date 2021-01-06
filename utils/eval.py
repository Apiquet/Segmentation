#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to evaluation segmentation module
"""

import cv2
from glob import glob
import imageio
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import random
import tensorflow as tf
from tqdm import tqdm


nb_colors = 100
COLORS = [(random.randint(50, 200),
           random.randint(50, 200),
           random.randint(50, 200)) for i in range(nb_colors)]


def pltPredOnImg(model: tf.keras.Model, image_path: str, nb_classes: int,
                 input_shape=(300, 300), save_path=None):
    """
    Method to show segmentation map in RGB image

    Args:
        - Segmentation model
        - image path
        - (int) number of classes
    """
    fig = plt.figure(figsize=(8, 8))
    image = np.array(Image.open(image_path))
    origin_width, origin_height, _ = image.shape
    image_res = cv2.resize(image, input_shape,
                           interpolation=cv2.INTER_NEAREST)
    tf_image = tf.expand_dims(
        tf.convert_to_tensor(image_res, dtype=tf.float32)/255., axis=0)
    seg_map = model(tf_image)
    seg_map = tf.reshape(seg_map, shape=seg_map.shape[1:])
    seg_map_max = np.argmax(seg_map, axis=2)

    seg_img = np.zeros(
        (seg_map_max.shape[0], seg_map_max.shape[1], 3)).astype('float')

    for i in range(nb_classes):
        idx_to_update = (seg_map_max == i)
        seg_img[:, :, 0] += idx_to_update*COLORS[i][0]
        seg_img[:, :, 1] += idx_to_update*COLORS[i][1]
        seg_img[:, :, 2] += idx_to_update*COLORS[i][2]

    seg_img = cv2.resize(seg_img, (origin_height, origin_width),
                         interpolation=cv2.INTER_NEAREST)

    plt.imshow(seg_img/255.)
    plt.imshow(image/255., 'jet', interpolation='none', alpha=0.2)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

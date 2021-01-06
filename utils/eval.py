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


def pltPredOnImg(model: tf.keras.Model, image_path: str,
                 input_shape=(300, 300), save_path=None):
    """
    Method to infer a segmentation model on an image
    Display overlap original image + segmentation map

    Args:
        - Segmentation model
        - image path
        - (int) number of classes
        - (tuple) input_shape: model input shape (width, height)
        - (str) path to save image result
    """
    fig = plt.figure(figsize=(8, 8))
    image = np.array(Image.open(image_path))
    origin_width, origin_height, _ = image.shape
    image_res = cv2.resize(image, input_shape,
                           interpolation=cv2.INTER_NEAREST)
    tf_image = tf.expand_dims(
        tf.convert_to_tensor(image_res, dtype=tf.float32)/255., axis=0)
    seg_map = model(tf_image)
    n_classes = seg_map.shape[-1]
    seg_map = tf.reshape(seg_map, shape=seg_map.shape[1:])
    seg_map_max = np.argmax(seg_map, axis=2)

    seg_img = np.zeros(
        (seg_map_max.shape[0], seg_map_max.shape[1], 3)).astype('float')

    for i in range(n_classes):
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


def pltPredOnVideo(model, video_path: str, out_gif: str, start_idx: int = 0,
                   end_idx: int = -1, skip: int = 1, resize: tuple = None,
                   fps: int = 30, input_shape: tuple = (300, 300)):
    """
    Method to infer a segmenation model on a MP4 video
    Create a gif with overlap original image + segmentation map

    Args:
        - Segmentation model
        - (str) video path (MP4)
        - (str) out path (.gif file)
        - (int) start frame idx, default is 0
        - (int) end frame idx, default is -1
        - (int) skip: idx%skip != 0 is skipped
        - (tuple) resize: target resolution for the gif
        - (int) fps: fps of the output gif
        - (tuple) input_shape: model input shape (width, height)
    """
    cap = cv2.VideoCapture(video_path)
    imgs = []
    i = 0
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_idx != -1:
        number_of_frame = end_idx
    for _ in tqdm(range(number_of_frame)):
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if i <= start_idx:
            continue
        elif end_idx >= 0 and i > end_idx:
            break
        if i % skip != 0:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if resize:
            img.thumbnail(resize, Image.ANTIALIAS)
        image = np.array(img)
        origin_width, origin_height, _ = image.shape
        image_res = cv2.resize(image, input_shape,
                               interpolation=cv2.INTER_LINEAR)
        tf_image = tf.expand_dims(
            tf.convert_to_tensor(image_res, dtype=tf.float32)/255., axis=0)
        seg_map = model(tf_image)
        n_classes = seg_map.shape[-1]
        seg_map = tf.reshape(seg_map, shape=seg_map.shape[1:])
        seg_map_max = np.argmax(seg_map, axis=2)

        seg_img = np.zeros(
            (seg_map_max.shape[0], seg_map_max.shape[1], 3)).astype('float')

        for n_class in range(1, n_classes):
            idx_to_update = (seg_map_max == n_class)
            seg_img[:, :, 0] += idx_to_update*COLORS[n_class][0]
            seg_img[:, :, 1] += idx_to_update*COLORS[n_class][1]
            seg_img[:, :, 2] += idx_to_update*COLORS[n_class][2]

        seg_img = cv2.resize(seg_img, (origin_height, origin_width),
                             interpolation=cv2.INTER_LINEAR)
        seg_pil = Image.fromarray(seg_img.astype('uint8'), 'RGB')
        datas = seg_pil.getdata()
        origin_data = img.getdata()

        newData = []
        for idx, item in enumerate(datas):
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append(origin_data[idx])
            else:
                newData.append(item)

        seg_pil.putdata(newData)

        imgs.append(
            Image.blend(img,
                        seg_pil, alpha=0.7))
    imgs[0].save(out_gif, format='GIF',
                 append_images=imgs[1:],
                 save_all=True, loop=0)

    gif = imageio.mimread(out_gif)
    imageio.mimsave(out_gif, gif, fps=fps)

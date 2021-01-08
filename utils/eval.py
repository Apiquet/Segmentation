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


def get_seg_map_to_display(model: tf.keras.Model, img: Image,
                           classes: dict, input_shape: tuple = (300, 300),
                           legend_size=None, legend_xpos=None,
                           legend_ypos=None, verbose=False, classes_seen=[]):
    """
    Function to infer a segmentation model on an image
    Return PIL image as overlay between segmentation map and
    input image and the list of seen classes

    Args:
        - Segmentation model
        - (PIL.Image) img
        - (dict) possible classes in images
        - (tuple) input_shape: model input shape (width, height)
        - (int) legend_size: factor to multiply legend sized calculated
        - (int) legend_xpos: new legend x pos wanted
        - (int) legend_ypos: new legend y pos wanted
        - (bool) verbose: to print the current and new legend parameters
        - (list) classes_seen: classes to print on image (if other classes are
                 seen, they will be added to the list)
    """
    image = np.array(img)
    origin_height, origin_width, _ = image.shape
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
        if np.sum(idx_to_update) != 0 and n_class not in classes_seen:
            classes_seen.append(n_class)
        seg_img[:, :, 0] += idx_to_update*COLORS[n_class][0]
        seg_img[:, :, 1] += idx_to_update*COLORS[n_class][1]
        seg_img[:, :, 2] += idx_to_update*COLORS[n_class][2]

    seg_img = cv2.resize(seg_img, (origin_width, origin_height),
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
    overlap = Image.blend(img, seg_pil, alpha=0.7)

    draw = ImageDraw.Draw(overlap)
    line_width = int(0.0002 * origin_width * origin_height)
    point = (origin_width-0.4*origin_width,
             int(0.1*origin_height))
    if legend_size is not None or legend_xpos is not None or\
            legend_ypos is not None:
        if verbose:
            print("Calculated legend: size {}, (x, y) pos {}".format(
                line_width, point))
        if legend_size is not None:
            line_width = int(line_width*legend_size)
        if legend_xpos is not None:
            point_list = list(point)
            point_list[0] = int(legend_xpos)
            point = tuple(point_list)
        if legend_ypos is not None:
            point_list = list(point)
            point_list[1] = int(legend_ypos)
            point = tuple(point_list)
        if verbose:
            print("New legend: size {}, (x, y) pos {}".format(
                line_width*legend_size, point))

    font = ImageFont.truetype("arial.ttf", line_width)
    for i, el in enumerate(classes_seen):
        if el in classes:
            draw.text((point[0], point[1]*i),
                      classes[el],
                      fill=(COLORS[el][0], COLORS[el][1],
                            COLORS[el][2], 0), font=font)
    return overlap, classes_seen


def pltPredOnImg(model: tf.keras.Model, image_path: str, classes: dict,
                 input_shape=(300, 300), save_path=None, legend_size=None,
                 legend_xpos=None, legend_ypos=None):
    """
    Function to infer a segmentation model on an image
    Display overlap original image + segmentation map

    Args:
        - Segmentation model
        - image path
        - (int) number of classes
        - (tuple) input_shape: model input shape (width, height)
        - (str) path to save image result
        - (int) legend_size: factor to multiply legend sized calculated
        - (int) legend_xpos: new legend x pos wanted
        - (int) legend_ypos: new legend y pos wanted
    """
    fig = plt.figure(figsize=(8, 8))
    img = Image.open(image_path)
    verbose = 0
    if legend_size is not None or legend_xpos is not None or\
            legend_ypos is not None:
        verbose = 1

    overlap, _ = get_seg_map_to_display(model, img, classes, input_shape,
                                        legend_size=legend_size,
                                        legend_xpos=legend_xpos,
                                        legend_ypos=legend_ypos,
                                        verbose=verbose)
    plt.imshow(overlap)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def pltPredOnVideo(model, video_path: str, out_gif: str, classes: dict,
                   start_idx: int = 0, end_idx: int = -1, skip: int = 1,
                   resize: tuple = None, fps: int = 30,
                   input_shape: tuple = (300, 300), legend_size=None,
                   legend_xpos=None, legend_ypos=None):
    """
    Function to infer a segmenation model on a MP4 video
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
        - (int) legend_size: factor to multiply legend sized calculated
        - (int) legend_xpos: new legend x pos wanted
        - (int) legend_ypos: new legend y pos wanted
    """
    cap = cv2.VideoCapture(video_path)
    imgs = []
    i = 0
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_idx != -1:
        number_of_frame = end_idx
    verbose = True
    classes_seen = []
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

        overlap, classes_seen = get_seg_map_to_display(
            model, img, classes, input_shape,
            legend_size=legend_size,
            legend_xpos=legend_xpos,
            legend_ypos=legend_ypos,
            verbose=verbose,
            classes_seen=classes_seen)
        verbose = False
        imgs.append(overlap)

    imgs[0].save(out_gif, format='GIF',
                 append_images=imgs[1:],
                 save_all=True, loop=0)

    gif = imageio.mimread(out_gif)
    imageio.mimsave(out_gif, gif, fps=fps)

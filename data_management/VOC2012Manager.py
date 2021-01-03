#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pascal VOC2012 dataset manager for segmentation class
"""

import cv2
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm


class VOC2012Manager():

    def __init__(self, input_shape=(300, 300, 3), floatType=32):
        super(VOC2012Manager, self).__init__()
        if floatType == 32:
            self.floatType = tf.float32
        elif floatType == 16:
            self.floatType = tf.float16
        else:
            raise Exception('floatType should be either 32 or 16')

        # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html
        self.labels_to_name = {
            0: "background", 255: "unlabelled", 1: "aeroplane", 2: "bicycle",
            3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car", 8: "cat",
            9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse",
            14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep",
            18: "sofa", 19: "train", 20: "tv-monitor"
        }
        self.width, self.height, self.channels = input_shape

    def load_data(self, VOC2012_path: str, width_height: tuple):
        jpeg_path = VOC2012_path + "VOC2012/JPEGImages/"
        segmentation_path = VOC2012_path + "VOC2012/SegmentationClass/"
        annotations_path = glob(segmentation_path + "*")
        filenames_png = []
        original_shapes = []

        print("\nLoading images...")
        images = []
        for el in tqdm(annotations_path, position=0, leave=True):
            filename = os.path.basename(el)
            filenames_png.append(filename)
            el = jpeg_path + filename.replace(".png", ".jpg")
            image = Image.open(el)
            original_shapes.append(image.size)
            image = np.array(image)
            image = cv2.resize(image, width_height,
                               interpolation=cv2.INTER_NEAREST)
            images.append(tf.convert_to_tensor(image, dtype=tf.float32))

        print("\nLoading annotations...")
        annotations = []
        for el in tqdm(annotations_path, position=0, leave=True):
            annotation = Image.open(el)
            annotation = np.array(annotation)
            annotation[annotation > 20] = 0
            annotation = cv2.resize(annotation, width_height,
                                    interpolation=cv2.INTER_NEAREST)
            annotations.append(
                tf.expand_dims(
                    tf.convert_to_tensor(annotation, dtype=tf.int8), 2))

        print("\nConvert to tensor...")
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        annotations = tf.convert_to_tensor(annotations, dtype=tf.int8)
        print("\nDone")
        print("Images shape: {}, annotations shape: {}".format(
            images.shape, annotations.shape))
        return images, annotations, filenames_png, original_shapes

    def random_choice(self, inputs, n_samples, seed):
        """
        With replacement.
        Params:
          inputs (Tensor): Shape [n_states, n_features]
          n_samples (int): The number of random samples to take.
        Returns:
          sampled_inputs (Tensor): Shape [n_samples, n_features]
        """
        # (1, n_states) since multinomial requires 2D logits.
        uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)
        tf.random.set_seed(seed)
        ind = tf.random.categorical(uniform_log_prob, n_samples)
        ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

        return tf.gather(inputs, ind, name="random_choice")

    def visualize_data(self, images: tf.Tensor, annotations: tf.Tensor,
                       original_shapes=None, n_samples=5, seed=42):
        images_toprint = self.random_choice(images, n_samples, seed)
        annotations_toprint = self.random_choice(annotations, n_samples,
                                                 seed)
        if original_shapes is not None:
            original_shapes = self.random_choice(original_shapes, n_samples,
                                                 seed)

        fig, axs = plt.subplots(n_samples, 2, figsize=(13, 25), facecolor='w',
                                edgecolor='k')
        fig.subplots_adjust(hspace=.5, wspace=.001)
        axs = axs.ravel()
        subplot_idx = 0
        for i in range(0, n_samples):
            image = tf.cast(images_toprint[i], dtype=tf.uint8).numpy()
            if original_shapes is not None:
                image = cv2.resize(image, tuple(original_shapes[i].numpy()),
                                   interpolation=cv2.INTER_LINEAR)

            annotation = annotations_toprint[i].numpy()
            if original_shapes is not None:
                annotation = cv2.resize(annotation,
                                        tuple(original_shapes[i].numpy()),
                                        interpolation=cv2.INTER_NEAREST)
            classes = np.unique(annotation)
            title = "Classes found: "
            for el in classes:
                if el != 0:
                    title += self.labels_to_name[el] + ' '

            axs[subplot_idx].imshow(image)
            axs[subplot_idx+1].imshow(annotation)
            axs[subplot_idx+1].set_title(title)
            axs[subplot_idx].set_xticks([])
            axs[subplot_idx+1].set_yticks([])
            subplot_idx += 2

    def prepare_data(self, images: tf.Tensor, annotations: tf.Tensor,
                     n_classes: int):
        # normalize images
        images = images / 255.

        gt_annotations = []
        print("Reshape gt from (300, 300, 1) to (300, 300, n_classes)")
        for n, annotation in enumerate(tqdm(annotations)):
            stack_list = []
            # Reshape segmentation masks
            for i in range(n_classes):
                mask = tf.equal(annotation[:, :, 0],
                                tf.constant(i, dtype=tf.int8))
                stack_list.append(tf.cast(mask, dtype=tf.int16))

            gt = tf.stack(stack_list, axis=2)
            gt_annotations.append(gt)
        return images, tf.convert_to_tensor(gt_annotations, dtype=tf.int8)

    def load_and_prepare_data(self, VOC2012_path: str, width_height: tuple,
                              n_classes=21, n_samples_to_show=0, seed=42):
        images, annotations, filenames_png, original_shapes = \
            self.load_data(VOC2012_path, width_height)
        images_normalized, gt_annotations = \
            self.prepare_data(images, annotations, n_classes)
        print("Images shape: {}, annotations shape: {}".format(
            images_normalized.shape, gt_annotations.shape))
        if n_samples_to_show > 0:
            print("Examples:")
            self.visualize_data(images, annotations, original_shapes,
                                n_samples_to_show, seed)
        return images_normalized, gt_annotations,\
            filenames_png, original_shapes

    def tensorf2TFData(self, images: tf.Tensor, annotations: tf.Tensor,
                       train_ratio=0.7, batch_size=32):
        dataset_size = images.shape[0]
        nb_train_samples = int(train_ratio * dataset_size)

        full_dataset = tf.data.Dataset.from_tensor_slices(
            (images, annotations)).shuffle(1024)
        train_dataset = full_dataset.take(
            nb_train_samples).batch(batch_size).prefetch(-1)
        val_dataset = full_dataset.skip(nb_train_samples)
        return train_dataset, val_dataset

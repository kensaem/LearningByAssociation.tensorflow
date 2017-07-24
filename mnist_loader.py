import os
import glob
import collections
import cv2
import numpy as np
import tensorflow as tf
import shutil
import math
from loader import *
from tensorflow.examples.tutorials.mnist import input_data


class MnistLoader(Loader):

    def __init__(self, data_path, default_batch_size, image_info, dataset="train"):
        self.mnist = input_data.read_data_sets(data_path, one_hot=True)
        self.dataset = dataset
        super().__init__(data_path, default_batch_size, image_info)

    def load_data(self):
        # Load data from mnist
        print("...Loading from %s" % self.data_path)

        if self.dataset == "train":
            images = self.mnist.train.images
            labels = self.mnist.train.labels
        elif self.dataset == "validation":
            images = self.mnist.validation.images
            labels = self.mnist.validation.labels
        elif self.dataset == "test":
            images = self.mnist.test.images
            labels = self.mnist.test.labels
        else:
            print("Invalid target dataset for MnistLoader")
            exit(1)

        for idx in range(len(images)):
            self.data.append(BatchTuple(
                images=np.reshape(
                    images[idx]*255,
                    (self.image_info.height, self.image_info.width, self.image_info.channel)),
                labels=np.argmax(labels[idx]))
            )

        print("\tTotal number of data = %d" % len(self.data))
        print("...Loading done.")
        self.reset()
        return

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.default_batch_size

        if (self.cur_idx + batch_size) > len(self.data):
            self.reset()
            return None

        batch = BatchTuple(
            images=np.zeros(
                dtype=np.uint8,
                shape=[batch_size, self.image_info.height, self.image_info.width, self.image_info.channel]
            ),
            labels=np.zeros(dtype=np.int32, shape=[batch_size])
        )

        for idx in range(batch_size):
            single_data = self.data[self.perm_idx[self.cur_idx + idx]]
            batch.images[idx, :, :, :] = single_data.images
            batch.labels[idx] = single_data.labels
        self.cur_idx += batch_size

        return batch


import os
import glob
import collections
import cv2
import numpy as np
import tensorflow as tf

BatchTuple = collections.namedtuple("BatchTuple", ['images', 'labels'])


class Loader:
    RawDataTuple = collections.namedtuple("RawDataTuple", ['path', 'label'])

    def __init__(self, data_path, default_batch_size):
        self.sess = tf.Session()
        self.image_info = {
            'width': 32,
            'height': 32,
            'channel': 3,
        }
        self.data = []
        self.default_batch_size = default_batch_size
        self.cur_idx = 0
        self.perm_idx = []
        self.epoch_counter = 0

        # Load data from directory
        print("...Loading from %s" % data_path)
        dir_name_list = os.listdir(data_path)
        for dir_name in dir_name_list:
            dir_path = os.path.join(data_path, dir_name)
            file_name_list = os.listdir(dir_path)
            print("\tNumber of files in %s = %d" % (dir_name, len(file_name_list)))
            for file_name in file_name_list:
                file_path = os.path.join(dir_path, file_name)
                self.data.append(self.RawDataTuple(path=file_path, label=int(dir_name)))
        print("\tTotal number of data = %d" % len(self.data))
        print("...Loading done.")
        self.reset()
        return

    def reset(self):
        self.cur_idx = 0
        self.perm_idx = np.random.permutation(len(self.data))
        self.epoch_counter += 1
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
                shape=[batch_size, self.image_info['height'], self.image_info['width'], self.image_info['channel']]
            ),
            labels=np.zeros(dtype=np.int32, shape=[batch_size])
        )

        for idx in range(batch_size):
            single_data = self.data[self.perm_idx[self.cur_idx + idx]]
            image = cv2.imread(single_data.path)
            batch.images[idx, :, :, :] = image
            batch.labels[idx] = single_data.label
        self.cur_idx += batch_size

        return batch


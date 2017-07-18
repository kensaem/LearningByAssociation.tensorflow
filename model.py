# Implementation of "Learning by Association / a versatile semi-supervised training method for neural networks"
# https://arxiv.org/abs/1706.00909
# Author : kensaem@gmail.com

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(100)
print(batch_xs.shape)
print(batch_ys.shape)

def weight_variable(shape, name=None):
    return tf.get_variable(
        name + "/weight",
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=.0, stddev=.1),
        dtype=tf.float32
    )


def bias_variable(shape, name=None):
    return tf.get_variable(
        name + "/bias",
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=.0, stddev=.0),
        dtype=tf.float32
    )


def conv_block(input_tensor, output_channel, layer_name, k_size=3):
    input_channel = input_tensor.get_shape().as_list()[-1]
    output_tensor = input_tensor

    with tf.variable_scope(layer_name):
        w_conv = weight_variable([k_size, k_size, input_channel, output_channel], name=layer_name)
        b_conv = bias_variable([output_channel], name=layer_name)
        output_tensor = tf.nn.conv2d(output_tensor, w_conv, strides=[1, 1, 1, 1], padding='SAME')
        output_tensor = tf.nn.bias_add(output_tensor, b_conv)
        output_tensor = tf.nn.elu(output_tensor)
        tf.summary.histogram(layer_name, output_tensor)

    return output_tensor


class Model:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')

        self.labeled_image_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[self.batch_size, 32, 32, 3],
            name='labeled_image_ph')

        self.label_ph = tf.placeholder(
            dtype=tf.int64,
            shape=[self.batch_size],
            name='label_ph')

        self.unlabeled_image_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[self.batch_size, 32, 32, 3],
            name='unlabeled_image_ph')

        self.labeled_output_t, self.labeled_output_cls = self.build_model(self.labeled_image_ph)
        self.unlabeled_output_t, self.unlabeled_output_cls = self.build_model(self.unlabeled_image_ph, reuse=True)

        self.epsilon = 1e-8
        self.walker_loss = self.build_walker_loss(self.labeled_output_t, self.unlabeled_output_t, self.label_ph)
        self.visit_loss = self.build_visit_loss(self.labeled_output_t, self.unlabeled_output_t)
        self.cls_loss = self.build_cls_loss(self.labeled_output_cls, self.label_ph)

        self.loss = self.walker_loss + self.visit_loss + self.cls_loss

        print(self.loss)
        exit(1)

        self.pred_label = tf.arg_max(tf.nn.softmax(self.labeled_output_cls), 1)

        optimizer = tf.train.AdamOptimizer
        self.train_op = optimizer(self.lr_placeholder).minimize(
            self.loss,
            global_step=self.global_step,
        )

        self.conf_matrix = tf.confusion_matrix(self.label_ph, self.pred_label, num_classes=10)
        self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred_label, self.label_ph)), axis=0)
        print(self.cls_loss)
        print(self.pred_label)

        return

    def build_model(self, input_t, name="base_model", reuse=False):
        layers_size = [2, 2, 2]

        output_t = input_t
        output_t = tf.div(tf.to_float(output_t), 255.0, name="input_image_float")

        with tf.variable_scope(name, reuse=reuse):

            for layers_idx in range(len(layers_size)):

                for idx in range(layers_size[0]):
                    output_t = conv_block(
                        input_tensor=output_t,
                        output_channel=32 * (2 ** layers_idx),
                        layer_name="layer_" + str(layers_idx) + "_" + str(idx + 1),
                    )
                with tf.variable_scope("layer" + str(layers_idx) + "_pooling"):
                    output_t = tf.nn.max_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                              data_format='NHWC')

            # input size 1 w/ channel 512 => fc layer
            shape = output_t.shape[1] * output_t.shape[2] * output_t.shape[3]
            print(shape.value)
            output_t = tf.reshape(output_t, [-1, int(shape.value)])

            with tf.variable_scope("fc_1"):
                w_fc1 = weight_variable([shape, 128], name="fc_1")
                b_fc1 = bias_variable([128], name="fc_1")
                output_t = tf.nn.bias_add(tf.matmul(output_t, w_fc1), b_fc1)
                output_t = tf.nn.elu(output_t)
                tf.summary.histogram("fc_1", output_t)

            with tf.variable_scope("fc_2"):
                w_fc2 = weight_variable([128, 10], name="fc_2")
                b_fc2 = bias_variable([10], name="fc_2")
                output_cls = tf.matmul(output_t, w_fc2) + b_fc2
                tf.summary.histogram("fc_2", output_cls)

            print("output layer for embedding =", output_t)
            print("output layer for classification =", output_cls)
        return output_t, output_cls

    def build_walker_loss(self, labeled_output_t, unlabeled_output_t, label):
        similarity = tf.matmul(labeled_output_t, unlabeled_output_t, transpose_b=True)
        transition_prob_to_unlabled = tf.nn.softmax(similarity, dim=1)
        transition_prob_to_labled = tf.nn.softmax(tf.transpose(similarity), dim=1)

        roundtrip_prob = tf.matmul(transition_prob_to_unlabled, transition_prob_to_labled)
        print(roundtrip_prob)

        label = tf.reshape(label, [-1, 1])
        target_distribution = tf.to_float(tf.equal(label, tf.transpose(label)))
        num_class = tf.reduce_sum(target_distribution, axis=1, keep_dims=True)
        target_distribution /= num_class
        print(target_distribution)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target_distribution,
            logits=tf.log(self.epsilon + roundtrip_prob),
            scope="walker_loss"
        )
        print(loss)
        return loss

    def build_visit_loss(self, labeled_output_t, unlabeled_output_t):
        similarity = tf.matmul(labeled_output_t, unlabeled_output_t, transpose_b=True)
        transition_prob_to_unlabled = tf.nn.softmax(similarity, dim=1)

        print(transition_prob_to_unlabled)
        visit_prob = tf.reduce_mean(tf.transpose(transition_prob_to_unlabled), axis=1)
        print(visit_prob)
        target_distribution = tf.constant(1.0/visit_prob.shape[0].value, shape=visit_prob.shape)
        print(target_distribution)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target_distribution,
            logits=tf.log(self.epsilon + visit_prob),
            scope="visit_loss",
        )
        print(loss)
        return loss

    def build_cls_loss(self, output_cls, label):
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=label,
            logits=output_cls,
            scope="classification_loss"
        )

        print(loss)
        return loss

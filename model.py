# Implementation of "Learning by Association / a versatile semi-supervised training method for neural networks"
# https://arxiv.org/abs/1706.00909
# Author : kensaem@gmail.com

import tensorflow as tf
import collections


def weight_variable(shape, name=None):
    return tf.get_variable(
        name + "/weight",
        shape=shape,
        initializer=tf.truncated_normal_initializer(mean=.0, stddev=2e-2),
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

    def __init__(self, batch_size, image_info):
        self.batch_size = batch_size

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.lr_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate')

        self.image_info = image_info

        self.labeled_image_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[self.batch_size, self.image_info.height, self.image_info.width, self.image_info.channel],
            name='labeled_image_ph')

        self.label_ph = tf.placeholder(
            dtype=tf.int64,
            shape=[self.batch_size],
            name='label_ph')

        self.unlabeled_image_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[self.batch_size, self.image_info.height, self.image_info.width, self.image_info.channel],
            name='unlabeled_image_ph')

        model_builder = self.build_vgg_model
        # model_builder = self.build_mnist_model

        self.labeled_output_t, self.labeled_output_cls = model_builder(self.labeled_image_ph)
        self.unlabeled_output_t, self.unlabeled_output_cls = model_builder(self.unlabeled_image_ph, reuse=True)

        self.epsilon = 1e-8
        self.walker_loss = self.build_walker_loss(self.labeled_output_t, self.unlabeled_output_t, self.label_ph)
        self.visit_loss = self.build_visit_loss(self.labeled_output_t, self.unlabeled_output_t)
        self.cls_loss = self.build_cls_loss(self.labeled_output_cls, self.label_ph)

        visit_weight = 0.25  # 0.25 for cifar-10, 1.0 for MNIST
        self.loss = self.walker_loss + visit_weight * self.visit_loss + self.cls_loss

        # L2 regularization
        variables = tf.trainable_variables()
        self.loss += tf.add_n([tf.nn.l2_loss(v) for v in variables if 'weight' in v.name]) * 1e-4

        self.pred_label = tf.arg_max(tf.nn.softmax(self.labeled_output_cls), 1)

        optimizer = tf.train.AdamOptimizer
        self.train_op = optimizer(self.lr_placeholder).minimize(
            self.loss,
            global_step=self.global_step,
        )

        self.conf_matrix = tf.confusion_matrix(self.label_ph, self.pred_label, num_classes=10)
        self.correct_count = tf.reduce_sum(tf.to_float(tf.equal(self.pred_label, self.label_ph)), axis=0)

        return

    def build_mnist_model(self, input_t, name="base_model", reuse=False):
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
            output_t = tf.reshape(output_t, [-1, int(shape.value)])

            with tf.variable_scope("fc_1"):
                w_fc1 = weight_variable([shape, 128], name="fc_1")
                b_fc1 = bias_variable([128], name="fc_1")
                output_t = tf.nn.bias_add(tf.matmul(output_t, w_fc1), b_fc1)
                output_embed = output_t
                output_t = tf.nn.elu(output_t)
                tf.summary.histogram("fc_1", output_t)

            with tf.variable_scope("fc_2"):
                w_fc2 = weight_variable([128, 10], name="fc_2")
                b_fc2 = bias_variable([10], name="fc_2")
                output_cls = tf.matmul(output_t, w_fc2) + b_fc2
                tf.summary.histogram("fc_2", output_cls)

            print("output layer for embedding =", output_embed)
            print("output layer for classification =", output_cls)
        return output_embed, output_cls

    def build_vgg_model(self, input_t, name="vgg_model", reuse=False):
        # layers_size = [1, 1, 2, 2, 2]  #vgg11
        layers_size = [2, 2, 3, 3, 3]  #vgg16

        with tf.variable_scope(name, reuse=reuse):

            output_t = input_t
            output_t = tf.div(tf.to_float(output_t), 255.0, name="input_image_float")

            # input size 32
            for idx in range(layers_size[0]):
                output_t = conv_block(
                    input_tensor=output_t,
                    output_channel=64,
                    layer_name="layer1_"+str(idx+1),
                )
            with tf.variable_scope("layer1_pooling"):
                output_t = tf.nn.max_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
            # input size 16
            for idx in range(layers_size[1]):
                output_t = conv_block(
                    input_tensor=output_t,
                    output_channel=128,
                    layer_name="layer2_"+str(idx+1),
                )
            with tf.variable_scope("layer2_pooling"):
                output_t = tf.nn.max_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
            # input size 8
            for idx in range(layers_size[2]):
                output_t = conv_block(
                    input_tensor=output_t,
                    output_channel=256,
                    layer_name="layer3_"+str(idx+1),
                )
            with tf.variable_scope("layer3_pooling"):
                output_t = tf.nn.max_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
            # input size 4
            for idx in range(layers_size[3]):
                output_t = conv_block(
                    input_tensor=output_t,
                    output_channel=512,
                    layer_name="layer4_"+str(idx+1),
                )
            with tf.variable_scope("layer4_pooling"):
                output_t = tf.nn.max_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
            # input size 2
            for idx in range(layers_size[4]):
                output_t = conv_block(
                    input_tensor=output_t,
                    output_channel=512,
                    layer_name="layer5_"+str(idx+1),
                )
            with tf.variable_scope("layer5_pooling"):
                output_t = tf.nn.max_pool(output_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')

            # input size 1 w/ channel 512 => fc layer
            output_t = tf.squeeze(output_t, axis=[1, 2])
            with tf.variable_scope("fc_1"):
                w_fc1 = weight_variable([512, 128], name="fc_1")
                b_fc1 = bias_variable([128], name="fc_1")
                output_t = tf.matmul(output_t, w_fc1) + b_fc1
                output_embed = output_t
                output_t = tf.nn.elu(output_t)
                tf.summary.histogram("fc_1", output_t)

            with tf.variable_scope("fc_2"):
                w_fc2 = weight_variable([128, 10], name="fc_2")
                b_fc2 = bias_variable([10], name="fc_2")
                output_cls = tf.matmul(output_t, w_fc2) + b_fc2
                tf.summary.histogram("fc_2", output_cls)

            print("output layer for embedding =", output_embed)
            print("output layer for classification =", output_cls)

        return output_embed, output_cls

    def build_walker_loss(self, labeled_output_t, unlabeled_output_t, label):
        similarity = tf.matmul(labeled_output_t, unlabeled_output_t, transpose_b=True)
        transition_prob_to_unlabeled = tf.nn.softmax(similarity, dim=1, name="transition_prob_to_unlabeled")
        transition_prob_to_labeled = tf.nn.softmax(tf.transpose(similarity), dim=1, name="transition_prob_to_labeled")

        roundtrip_prob = tf.matmul(transition_prob_to_unlabeled, transition_prob_to_labeled, name="roundtrip_prob")
        print(roundtrip_prob)
        tf.summary.histogram("roundtrip_prob", roundtrip_prob)

        label = tf.reshape(label, [-1, 1])
        target_distribution = tf.to_float(tf.equal(label, tf.transpose(label)))
        num_class = tf.reduce_sum(target_distribution, axis=1, keep_dims=True)
        tf.summary.histogram("num_class", num_class)
        target_distribution = target_distribution / num_class

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target_distribution,
            logits=tf.log(self.epsilon + roundtrip_prob),
        )

        tf.summary.scalar("loss/walker", loss)
        return loss

    def build_visit_loss(self, labeled_output_t, unlabeled_output_t):
        similarity = tf.matmul(labeled_output_t, unlabeled_output_t, transpose_b=True)
        transition_prob_to_unlabeled = tf.nn.softmax(similarity, dim=1)

        visit_prob = tf.reduce_mean(transition_prob_to_unlabeled, axis=0, keep_dims=True)

        target_distribution = tf.constant(1.0/visit_prob.shape[1].value, shape=visit_prob.shape)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=target_distribution,
            logits=tf.log(self.epsilon + visit_prob),
        )

        tf.summary.scalar("loss/visit", loss)
        return loss

    def build_cls_loss(self, output_cls, label):
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=label,
            logits=output_cls,
        )
        tf.summary.scalar("loss/classification", loss)
        return loss

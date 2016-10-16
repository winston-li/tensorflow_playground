from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

import mnist_input

# Network Parameters
NUM_INPUT_XS = mnist_input.IMAGE_SIZE
NUM_INPUT_YS = mnist_input.IMAGE_SIZE
NUM_INPUT_FEATURES = mnist_input.IMAGE_PIXELS
NUM_OUTPUT = mnist_input.NUM_CLASSES
NUM_FC_1 = 100  # 1st fully-connected layer number of features
PATCH_SZ = 5  # convolution patch size 5x5
NUM_CONV_1 = 20  # convolution layer1 output channels
NUM_CONV_2 = 40  # convolution layer2 output channels


# Helper functions
def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1 / math.sqrt(shape[0]))
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def _conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def _nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = _weight_variable([input_dim, output_dim])
        with tf.name_scope('biases'):
            biases = _bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):
            preactivations = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations',
                                 preactivations)

        activations = act(preactivations,
                          'activations') if act != None else tf.identity(
                              preactivations, name='activations')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations


def _conv_layer(input_tensor,
                psize,
                in_channels,
                out_channels,
                layer_name,
                act=tf.nn.relu):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = _weight_variable(
                [psize, psize, in_channels, out_channels])
        with tf.name_scope('biases'):
            biases = _bias_variable([out_channels])
        with tf.name_scope('Wx_plus_b'):
            preactivations = _conv2d(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations',
                                 preactivations)

        activations = act(preactivations,
                          'activations') if act != None else tf.identity(
                              preactivations, name='activations')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations


def _maxpool_layer(input_tensor, ksize, layer_name):
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        activations = tf.nn.max_pool(
            input_tensor,
            ksize=[1, ksize, ksize, 1],
            strides=[1, ksize, ksize, 1],
            padding='SAME',
            name='max_pool')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations


def _build_model(input_tensor, dropout_keep_rate):
    # Convolutional Layer (5x5 patch size, 1 input channel, 32 output channels)
    x_image = tf.reshape(input_tensor, [-1, NUM_INPUT_XS, NUM_INPUT_YS, 1])
    layer_1 = _conv_layer(
        x_image, 5, 1, NUM_CONV_1,
        'layer1')  # outputs: NUM_CONV_1 * NUM_INPUT_XS * NUM_INPUT_YS

    # 2x2 Max Pooling Layer
    xs = NUM_INPUT_XS // 2
    ys = NUM_INPUT_YS // 2
    layer_2 = _maxpool_layer(layer_1, 2,
                             'layer2')  # outputs: NUM_CONV_1 * xs * ys

    # Convolutional Layer (5x5 patch size, 32 input channel, 64 output channels)
    layer_3 = _conv_layer(layer_2, 5, NUM_CONV_1, NUM_CONV_2,
                          'layer3')  # outputs: NUM_CONV_2 * xs * ys

    # 2x2 Max Pooling Layer
    xs = xs // 2
    ys = ys // 2
    layer_4 = _maxpool_layer(layer_3, 2,
                             'layer4')  # outputs: NUM_CONV_2 * xs * ys
    layer_4_flat = tf.reshape(layer_4, [-1, NUM_CONV_2 * xs * ys])

    # Full-Connected Layer with Dropout
    layer_5 = _nn_layer(layer_4_flat, NUM_CONV_2 * xs * ys, NUM_FC_1, 'layer5')
    layer_5_drop = tf.nn.dropout(
        layer_5, dropout_keep_rate, name='dropout_layer')

    # Output Layer
    output_layer = _nn_layer(
        layer_5_drop, NUM_FC_1, NUM_OUTPUT, 'output_layer', act=None)
    return output_layer


def inference(input_tensor, dropout_keep_rate):
    with tf.name_scope('Model'):
        logits = _build_model(input_tensor, dropout_keep_rate)
        pred = tf.nn.softmax(logits, name='pred')
        return pred, logits


# TF Model Graph placeholders
def placeholders():
    with tf.name_scope('Input'):
        images_ph = tf.placeholder(
            tf.float32, [None, NUM_INPUT_XS * NUM_INPUT_YS], name='images')
        labels_ph = tf.placeholder(
            tf.float32, [None, NUM_OUTPUT], name='labels')
        keep_rate_ph = tf.placeholder(tf.float32, name='dropout_keep_rate')
        return images_ph, labels_ph, keep_rate_ph


def accuracy(predictions, labels):
    with tf.name_scope('Accuracy'):
        accuracy = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(
            tf.cast(accuracy, tf.float32), name='accuracy')
        tf.scalar_summary('accuracy', accuracy)
        return accuracy


def loss(logits, labels):
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels),
            name='loss')
        tf.scalar_summary('loss', loss)
        #tf.add_to_collection('losses', loss)
        return loss


def avg_loss():
    with tf.name_scope('Loss'):
        avg_loss = tf.Variable(0.0, trainable=False, name='avg_loss')
        tf.scalar_summary('avg_loss', avg_loss)
        tf.add_to_collection('avg_loss', avg_loss)
        return avg_loss


def _add_loss_summaries():
    with tf.name_scope('Loss'):
        avg_loss = tf.get_collection('avg_loss')
        # compute the exponential moving average of all average losses
        ema_loss = tf.train.ExponentialMovingAverage(0.9, name='ema')
        ema_loss_op = ema_loss.apply([avg_loss[0]])
        tf.scalar_summary('ema_loss', ema_loss.average(avg_loss[0]))
        return ema_loss_op


'''
def _add_loss_summaries(total_loss):
    with tf.name_scope('Loss'):
        # compute the exponential moving average of all average losses
        ema_loss = tf.train.ExponentialMovingAverage(0.9, name='ema')
        ema_loss_op = ema_loss.apply([total_loss])
        tf.scalar_summary('ema_loss', ema_loss.average(total_loss))
        return ema_loss_op
'''


def training(total_loss, learning_rate, global_step):
    # Generate moving averages of loss and associated summaries.
    ema_loss_op = _add_loss_summaries()
    with tf.name_scope('Optimizer'):
        with tf.control_dependencies([ema_loss_op]):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(
                total_loss, global_step=global_step, name='train_op')
            return train_op

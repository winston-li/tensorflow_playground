from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

import drone_input

# Network Parameters
NUM_INPUT_XS = drone_input.IMAGE_WIDTH
NUM_INPUT_YS = drone_input.IMAGE_HEIGHT
NUM_INPUT_ZS = drone_input.IMAGE_DEPTH
NUM_INPUT_FEATURES = drone_input.IMAGE_PIXELS
NUM_OUTPUT = drone_input.NUM_CLASSES
NUM_FC_1 = 200  # 1st fully-connected layer number of features
PATCH_SZ = 4  # convolution patch size 5x5
NUM_CONV_1 = 32  # convolution layer1 output channels
NUM_CONV_2 = 32  # convolution layer2 output channels
NUM_CONV_3 = 32  # convolution layer3 output channels
NUM_CONV_4 = 32  # convolution layer4 output channels


# Helper functions
def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1 / math.sqrt(shape[0]))
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def _conv2d(x, W, padding='VALID'):
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
    # Convolutional Layer (4x4 patch size, 3 input channels, 32 output channels)
    x_image = tf.reshape(input_tensor, [-1, NUM_INPUT_XS, NUM_INPUT_YS, NUM_INPUT_ZS])
    layer_1 = _conv_layer(
        x_image, 4, NUM_INPUT_ZS, NUM_CONV_1,
        'layer1')  # outputs: NUM_CONV_1 * (NUM_INPUT_XS-3) * (NUM_INPUT_YS-3)

    # 2x2 Max Pooling Layer 
    # Note: maxpooling with 'SAME' padding, +1 for xs,ys ceiling value of division 
    xs = (NUM_INPUT_XS - 3 + 1) // 2
    ys = (NUM_INPUT_YS - 3 + 1) // 2 
    layer_2 = _maxpool_layer(layer_1, 2,
                             'layer2')  # outputs: NUM_CONV_1 * xs * ys

    # Convolutional Layer (4x4 patch size, 32 input channel, 32 output channels)
    layer_3 = _conv_layer(layer_2, 4, NUM_CONV_1, NUM_CONV_2,
                          'layer3')  # outputs: NUM_CONV_2 * (xs-3) * (ys-3)

    # 2x2 Max Pooling Layer
    xs = (xs - 3 + 1) // 2
    ys = (ys - 3 + 1) // 2
    layer_4 = _maxpool_layer(layer_3, 2,
                             'layer4')  # outputs: NUM_CONV_2 * xs * ys

    # Convolutional Layer (4x4 patch size, 32 input channel, 32 output channels)
    layer_5 = _conv_layer(layer_4, 4, NUM_CONV_2, NUM_CONV_3,
                          'layer5')  # outputs: NUM_CONV_3 * (xs-3) * (ys-3)

    # 2x2 Max Pooling Layer
    xs = (xs - 3 + 1) // 2
    ys = (ys - 3 + 1) // 2
    layer_6 = _maxpool_layer(layer_5, 2,
                             'layer6')  # outputs: NUM_CONV_3 * xs * ys


    # Convolutional Layer (4x4 patch size, 32 input channel, 32 output channels)
    layer_7 = _conv_layer(layer_6, 4, NUM_CONV_3, NUM_CONV_4,
                          'layer7')  # outputs: NUM_CONV_4 * (xs-3) * (ys-3)

    # 2x2 Max Pooling Layer
    xs = (xs - 3 + 1) // 2
    ys = (ys - 3 + 1) // 2
    layer_8 = _maxpool_layer(layer_7, 2,
                             'layer8')  # outputs: NUM_CONV_4 * xs * ys

    layer_8_flat = tf.reshape(layer_8, [-1, NUM_CONV_4 * xs * ys])

    # Full-Connected Layer with Dropout
    layer_9 = _nn_layer(layer_8_flat, NUM_CONV_4 * xs * ys, NUM_FC_1, 'layer9')
    layer_9_drop = tf.nn.dropout(
        layer_9, dropout_keep_rate, name='dropout_layer')

    # Output Layer
    output_layer = _nn_layer(
        layer_9_drop, NUM_FC_1, NUM_OUTPUT, 'output_layer', act=None)
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
            tf.float32, [None, NUM_INPUT_FEATURES], name='images')
        labels_ph = tf.placeholder(
            tf.float32, [None, NUM_OUTPUT], name='labels')
        keep_rate_ph = tf.placeholder(tf.float32, name='dropout_keep_rate')
        return images_ph, labels_ph, keep_rate_ph


def evaluation(predictions, labels):
    with tf.name_scope('Evaluation'):
        accuracy = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(
            tf.cast(accuracy, tf.float32), name='accuracy')
        tf.scalar_summary('Evaluation/accuracy', accuracy)
        error = 1.0 - accuracy
        tf.scalar_summary('Evaluation/error', error)
        return accuracy


def loss(logits, labels):
    with tf.name_scope('Loss'):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels),
            name='cross_entropy')
        #tf.scalar_summary('loss', loss) # use avg_loss & ema_loss instead of just the most recent mini-batch only
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')


def avg_loss():
    with tf.name_scope('Loss'):
        avg_loss = tf.Variable(0.0, trainable=False, name='avg_loss')
        tf.scalar_summary('Loss/total_loss (avg)', avg_loss)
        return avg_loss


def _add_loss_summaries(total_loss):
    with tf.name_scope('Loss'):
        # compute the exponential moving average of all average losses
        ema_loss = tf.train.ExponentialMovingAverage(0.9, name='ema_loss')
        losses = tf.get_collection('losses')
        ema_loss_op = ema_loss.apply(losses + [total_loss])
        for l in losses + [total_loss]:
            tf.scalar_summary(l.op.name + ' (raw)', l)
            tf.scalar_summary(l.op.name + ' (ema)', ema_loss.average(l))
        return ema_loss_op


def training(total_loss, learning_rate, global_step):
    # Generate moving averages of loss and associated summaries.
    ema_loss_op = _add_loss_summaries(total_loss)
    with tf.name_scope('Optimizer'):
        with tf.control_dependencies([ema_loss_op]):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(
                total_loss, global_step=global_step, name='train_op')

            # TODO: Track the moving averages of all trainable variables.    
            return train_op

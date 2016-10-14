from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 10
PRED_RUNS = 3


def _get_graph_tensors(graph_path, tensors):
    graph_def = tf.GraphDef()
    with open(graph_path, "rb") as f:
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, return_elements=tensors, name='')


# Load finalized model graph, and use it to predict new data
def pred(graph_path, data):
    tensors = ['Input/images:0', 'Input/dropout_keep_rate:0', 'Model/pred:0']
    images_ph, keep_prob_ph, pred = _get_graph_tensors(graph_path, tensors)
    with tf.Session() as sess:
        for i in range(PRED_RUNS):
            batch_xs, batch_ys = data.next_batch(BATCH_SIZE)
            predictions = sess.run(
                pred, feed_dict={images_ph: batch_xs,
                                 keep_prob_ph: 1.0})
            print('Round %d: predicts = %s ' % (i, np.argmax(predictions, 1)))
            print('Round %d: labels   = %s ' % (i, np.argmax(batch_ys, 1)))


def run():
    model_dir = os.path.join(os.getcwd(), 'models')
    graph_path = os.path.join(model_dir, 'model.pb')
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    pred(graph_path, mnist_data.test)


if __name__ == '__main__':
    run()

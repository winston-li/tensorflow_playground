from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 10
PRED_RUNS = 3


def _get_graph_nodes(graph_path, nodes):
    graph_def = tf.GraphDef()
    with open(graph_path, "rb") as f:
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, return_elements=nodes, name='')


# Load finalized model graph, and use it to predict new data
def pred(graph_path, data):
    nodes = ['input_data:0', 'dropout_keep_rate:0', 'Model/pred:0']
    x, keep_prob, pred = _get_graph_nodes(graph_path, nodes)
    with tf.Session() as sess:
        for i in range(PRED_RUNS):
            batch_xs, batch_ys = data.next_batch(BATCH_SIZE)
            predictions = sess.run(pred,
                                   feed_dict={x: batch_xs,
                                              keep_prob: 1.0})
            print('Round %d: predicts = %s ' % (i, np.argmax(predictions, 1)))
            print('Round %d: labels   = %s ' % (i, np.argmax(batch_ys, 1)))


def run():
    model_path = os.path.join(os.getcwd(), 'models')
    graph_path = os.path.join(model_path, 'model.pb')
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    pred(graph_path, mnist_data.test)


if __name__ == '__main__':
    run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100

def test(model_path, eval=False):
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    data = mnist_data.test if not eval else mnist_data.validation
    with tf.Session() as sess:    
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore from ", ckpt.model_checkpoint_path)
            persistenter = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            persistenter.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        
        total_batch = int(data.num_examples / BATCH_SIZE)
        elapsed_time = 0.0
        
        x = sess.graph.get_tensor_by_name('input_data:0')
        y_ = sess.graph.get_tensor_by_name('label_data:0')        
        keep_rate = sess.graph.get_tensor_by_name('dropout_keep_rate:0')            
        accuracy = sess.graph.get_tensor_by_name('Accuracy/accuracy:0')
        pred = sess.graph.get_tensor_by_name('Model/pred:0') # used to predict, not used in this test accuracy function
        
        acc = 0.0
        for i in range(total_batch):
            batch_xs, batch_ys = data.next_batch(BATCH_SIZE)
            start_time = time.time()
            value = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_rate: 1.0})
            elapsed_time += (time.time() - start_time)
            acc += value * BATCH_SIZE
        pred_accuracy = acc / data.num_examples
        print("Accuracy: {:.5f}, elipsed time: {:.5f} sec for {} samples".format(pred_accuracy, elapsed_time, data.num_examples))

if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(), 'models')
    print('Validation accuracy:')
    test(model_path, eval=True)
    print('Test accurary:')
    test(model_path, eval=False)


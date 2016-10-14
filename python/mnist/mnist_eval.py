from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time
import mnist_input

BATCH_SIZE = 100
MAX_STEPS = 100


def evaluate(model_dir, data_dir, dataset_type):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore from ", ckpt.model_checkpoint_path)
            persistenter = tf.train.import_meta_graph(
                ckpt.model_checkpoint_path + '.meta')
            persistenter.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        images_ph = sess.graph.get_tensor_by_name('Input/images:0')
        labels_ph = sess.graph.get_tensor_by_name('Input/labels:0')
        keep_rate_ph = sess.graph.get_tensor_by_name(
            'Input/dropout_keep_rate:0')
        accuracy = sess.graph.get_tensor_by_name('Accuracy/accuracy:0')

        images, labels = mnist_input.input_pipeline(
            data_dir, BATCH_SIZE, type=dataset_type)

        coord = tf.train.Coordinator()
        # Note: QueueRunner created in mnist_input.py
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        elapsed_time = 0.0
        acc = 0.0
        try:
            step = 0
            while step < MAX_STEPS and not coord.should_stop():
                images_r, labels_r = sess.run([images, labels])
                data_feed = {
                    images_ph: images_r,
                    labels_ph: labels_r,
                    keep_rate_ph: 1.0
                }

                start_time = time.time()
                value = sess.run(accuracy, feed_dict=data_feed)
                elapsed_time += (time.time() - start_time)
                acc += value * BATCH_SIZE
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done validation/test for %d samples' % (step * BATCH_SIZE))

        finally:
            # When done, ask the threads to stop
            coord.request_stop()

        coord.join(threads)
        pred_accuracy = acc / (step * BATCH_SIZE)
        print("Accuracy: %.5f, elipsed time: %.5f sec for %d samples" %
              (pred_accuracy, elapsed_time, step * BATCH_SIZE))


def run():
    data_dir = os.path.join(os.getcwd(), 'MNIST_data')
    model_dir = os.path.join(os.getcwd(), 'models')
    print('Validation accuracy:')
    evaluate(model_dir, data_dir, mnist_input.DataTypes.validation)
    print('Test accurary:')
    evaluate(model_dir, data_dir, mnist_input.DataTypes.test)


if __name__ == '__main__':
    run()

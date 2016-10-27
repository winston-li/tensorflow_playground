from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time
import drone_input
import numpy as np

BATCH_SIZE = 100
dataset_map = [' (train)', ' (validataion)', ' (test)', ' (all)']


def evaluate(model_dir, data_dir, log_dir, dataset_type):
    with tf.Session() as sess:
        # Restore Graph
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
        accuracy = sess.graph.get_tensor_by_name('Evaluation/accuracy:0')
        images, _, _, _, labels, _, _ = drone_input.input_pipeline(
            data_dir, BATCH_SIZE, type=dataset_type, epochs=1)
        global_step = tf.get_collection(tf.GraphKeys.VARIABLES, 'global_step')[0]

        sess.run(tf.initialize_local_variables()) # for string_input_producer in input_pipeline

        acc_step = sess.run(global_step)
        print('accumulated step = %d' % acc_step)

        coord = tf.train.Coordinator()
        # Note: QueueRunner created in drone_input.py
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        elapsed_time = 0.0
        acc = 0.0
        count = 0
        try:
            while not coord.should_stop():
                images_r, labels_r = sess.run([images, labels])
                data_feed = {
                    images_ph: images_r,
                    labels_ph: labels_r,
                    keep_rate_ph: 1.0
                }

                start_time = time.time()
                value = sess.run(accuracy, feed_dict=data_feed)
                elapsed_time += (time.time() - start_time)
                acc += value * len(images_r)
                count += len(images_r)

        except tf.errors.OutOfRangeError:
            print('Done evaluation for %d samples' % count)

        finally:
            # When done, ask the threads to stop
            coord.request_stop()

        coord.join(threads)
        pred_accuracy = acc / count
        eval_writer = tf.train.SummaryWriter(log_dir + '/evaluation', sess.graph)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Evaluation/accuracy ' + dataset_map[dataset_type], simple_value=pred_accuracy), 
        ])
        eval_writer.add_summary(summary, acc_step)        
        print("Accuracy %s: %.5f, elipsed time: %.5f sec for %d samples" %
              (dataset_map[dataset_type], pred_accuracy, elapsed_time, count))



def run():
    data_dir = os.path.join(os.getcwd(), 'drone_data', 'tfrecord')
    model_dir = os.path.join(os.getcwd(), 'models')
    log_dir = os.path.join(os.getcwd(), 'logs')
    print('Validation accuracy:')
    evaluate(model_dir, data_dir, log_dir, drone_input.DataTypes.validation)
    print('Test accurary:')
    evaluate(model_dir, data_dir, log_dir, drone_input.DataTypes.test)
    print('Train accurary:')
    evaluate(model_dir, data_dir, log_dir, drone_input.DataTypes.train)


if __name__ == '__main__':
    run()

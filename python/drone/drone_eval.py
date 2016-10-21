from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time
import drone_input
import numpy as np

BATCH_SIZE = 100
# GS + TL + TR for followings 
TRAIN_DATA_SIZE = 24474 + 24913 + 23425  # 72812
VALIDATION_DATA_SIZE = 722 + 211 + 476   #  1409
TEST_DATA_SIZE = 25738 + 4083 + 4084     # 33905

dataset_map = [' (train)', ' (validataion)', ' (test)']


def evaluate(model_dir, data_dir, log_dir, dataset_type, max_steps):
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
            data_dir, BATCH_SIZE, type=dataset_type)
        global_step = tf.get_collection(tf.GraphKeys.VARIABLES, 'global_step')[0]

        acc_step = sess.run(global_step)
        print('accumulated step = %d' % acc_step)

        coord = tf.train.Coordinator()
        # Note: QueueRunner created in drone_input.py
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        elapsed_time = 0.0
        acc = 0.0
        try:
            step = 0
            while step < max_steps and not coord.should_stop():
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
        eval_writer = tf.train.SummaryWriter(log_dir + '/evaluation', sess.graph)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Evaluation/accuracy ' + dataset_map[dataset_type], simple_value=pred_accuracy), 
        ])
        eval_writer.add_summary(summary, acc_step)        
        print("Accuracy %s: %.5f, elipsed time: %.5f sec for %d samples" %
              (dataset_map[dataset_type], pred_accuracy, elapsed_time, step * BATCH_SIZE))



def run():
    data_dir = os.path.join(os.getcwd(), 'drone_data', 'tfrecord')
    model_dir = os.path.join(os.getcwd(), 'models')
    log_dir = os.path.join(os.getcwd(), 'logs')
    print('Validation accuracy:')
    evaluate(model_dir, data_dir, log_dir, drone_input.DataTypes.validation, VALIDATION_DATA_SIZE / BATCH_SIZE)
    print('Test accurary:')
    evaluate(model_dir, data_dir, log_dir, drone_input.DataTypes.test, TEST_DATA_SIZE / BATCH_SIZE)
    print('Train accurary:')
    evaluate(model_dir, data_dir, log_dir, drone_input.DataTypes.train, TRAIN_DATA_SIZE / BATCH_SIZE)


if __name__ == '__main__':
    run()

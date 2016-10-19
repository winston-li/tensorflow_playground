from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import drone_tfrecords as tfr

BATCH_SIZE = 3  # read 3 tfrecords per batch
MAX_STEPS = 1  # display how many batches per tfrecord file
NEW_SIZE = 101


def input_pipeline(filename, batch_size):
    #   ATTENTION:
    #   DO NOT set num_epochs in string_input_producer, otherwise uninitialized variable error occurs...
    #   To resolve this, need to initialize local variables, but it will make it difficult while restoring model checkpoints...
    filename_queue = tf.train.string_input_producer(
        [filename], name='string_input_producer')

    image, height, width, depth, label_id, label_txt, filename = tfr.read_and_decode_example(
        filename_queue)

    # Collect examples into batch_size batches.
    # (Internally uses a Queue.)
    # We run this in two threads to avoid being a bottleneck.
    images, heights, widths, depths, label_ids, label_txts, filenames = tf.train.batch(
        [image, height, width, depth, label_id, label_txt, filename],
        batch_size=batch_size,
        num_threads=2,
        capacity=1000 + 3 * batch_size,
        enqueue_many=False,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name='input_batching')
    return images, heights, widths, depths, label_ids, label_txts, filenames


def check_tfrs(filename, max_steps):
    with tf.Session() as sess:
        images, heights, widths, depths, label_ids, label_txts, filenames = input_pipeline(
            filename, BATCH_SIZE)

        coord = tf.train.Coordinator()
        # Note: QueueRunner created in drone_input.py
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while step < max_steps and not coord.should_stop():
                images_r, heights_r, widths_r, depths_r, label_ids_r, label_txts_r, filenames_r = sess.run(
                    [
                        images, heights, widths, depths, label_ids, label_txts,
                        filenames
                    ])

                for i in range(len(images_r)):
                    print('height: %d, width: %d, depth: %d' %
                          (heights_r[i], widths_r[i], depths_r[i]))
                    print('label_id: %d, label_txt: %s, filename: %s' %
                          (label_ids_r[i], label_txts_r[i], filenames_r[i]))
                    img = images_r[i].reshape(
                        [heights_r[i], widths_r[i], depths_r[i]])
                    plt.imshow(img)
                    plt.show()

                    re_image = tf.image.resize_images(images_r[i].reshape(
                        [heights_r[i], widths_r[i], depths_r[i]]), NEW_SIZE,
                                                      NEW_SIZE)
                    img2 = sess.run(re_image)
                    plt.imshow(img2.astype(np.uint8), interpolation='nearest')
                    plt.show()
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done validation/test for %d samples' % (step * BATCH_SIZE))

        finally:
            # When done, ask the threads to stop
            coord.request_stop()

        coord.join(threads)


def run():
    tfr_dir = os.path.join(os.getcwd(),
                           'drone_data',
                           'tfrecord', )
    for i in ['validation', 'train', 'test']:
        tfr_file_path = '%s/%s/*' % (tfr_dir, i)
        matching_files = tf.gfile.Glob(tfr_file_path)
        for filename in matching_files:
            print('TFRecord File: %s' % filename)
            check_tfrs(filename, MAX_STEPS)


if __name__ == '__main__':
    run()

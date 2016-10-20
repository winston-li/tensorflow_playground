from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import drone_tfrecords as tfr
import drone_input

BATCH_SIZE = 100  # read how many tfrecords per batch
MAX_STEPS = 1  # display how many batches
DISPLAY_PER_BATCH = 2 # disply images per batch
NEW_IMAGE_SIZE = 101 # for viewing resized images


def check_tfrs(data_dir, max_steps, batch_size, type):
    with tf.Session() as sess:
        images, heights, widths, depths, label_ids, label_txts, filenames = drone_input.input_pipeline(
            data_dir, batch_size, type)

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

                anchor = BATCH_SIZE // DISPLAY_PER_BATCH
                for i in range(len(images_r)):
                    if (i + 1) % anchor == 0:
                        print('height: %d, width: %d, depth: %d' %
                              (heights_r[i], widths_r[i], depths_r[i]))
                        print('label_id: %s, label_txt: %s, filename: %s' %
                              (label_ids_r[i], label_txts_r[i], filenames_r[i]))
                        #print(images_r[i].size)
                        img = images_r[i].reshape(
                            [heights_r[i], widths_r[i], depths_r[i]])
                        plt.imshow(img)
                        plt.show()

                        if (heights_r[i] != NEW_IMAGE_SIZE or widths_r[i] != NEW_IMAGE_SIZE):
                            re_image = tf.image.resize_images(images_r[i].reshape(
                                [heights_r[i], widths_r[i], depths_r[i]]),
                                                              NEW_IMAGE_SIZE, NEW_IMAGE_SIZE)
                            img2 = sess.run(re_image)
                            plt.imshow(np.around(img2).astype(np.uint8))
                            plt.show()
                step += 1

        except tf.errors.OutOfRangeError:
            print('Done check for %d samples' % (step * BATCH_SIZE))

        finally:
            # When done, ask the threads to stop
            coord.request_stop()

        coord.join(threads)


def run():
    tfr_dir = os.path.join(os.getcwd(), 'drone_data', 'tfrecord')
    check_tfrs(tfr_dir, MAX_STEPS, BATCH_SIZE,
               drone_input.DataTypes.validation)


if __name__ == '__main__':
    run()

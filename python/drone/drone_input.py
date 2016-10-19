from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path
import numpy as np
import drone_tfrecords as tfr

VALIDATION_DATA_SIZE = 5000
IMAGE_SIZE = 101
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 3

TFRECORDS_DIRS = ['train', 'validation', 'test']


class DataTypes:
    train, validation, test = range(3)


def input_pipeline(data_dir, batch_size, type=DataTypes.train):
    tfr_dir = os.path.join(data_dir, TFRECORDS_DIRS[type])
    tfr_file_path = '%s/*' % (tfr_dir)
    matching_files = tf.gfile.Glob(tfr_file_path)
    print(matching_files)

    #   ATTENTION:
    #   DO NOT set num_epochs in string_input_producer, otherwise uninitialized variable error occurs...
    #   To resolve this, need to initialize local variables, but it will make it difficult while restoring model checkpoints...
    filename_queue = tf.train.string_input_producer(
        matching_files, name='string_input_producer')

    image, height, width, depth, label_id, label_txt, filename = tfr.read_and_decode_example(
        filename_queue, NUM_CLASSES)

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

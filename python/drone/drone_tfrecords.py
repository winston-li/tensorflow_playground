from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
IMAGE_SIZE = 101
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 3

TFRECORDS_FILES = ['train.tfrecords', 'validation.tfrecords', 'test.tfrecords']

# Decode a TFRecord example
def read_and_decode_tfr(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/depth': tf.FixedLenFeature([], tf.int64),            
            'image/raw': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string)
        })

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    depth = tf.cast(features['image/depth'], tf.int32)
    image = tf.decode_raw(features['image/raw'], tf.uint8)

    image.set_shape([720*1280*3])
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label_id = tf.cast(features['image/class/label'], tf.int32)
    label_txt = features['image/class/text']
    filename = features['image/filename']

    return image, height, width, depth, label_id, label_txt, filename
    #return image, tf.one_hot(label, NUM_CLASSES)


class DataTypes:
    train, validation, test = range(3)


def input_pipeline(data_dir, batch_size, type=DataTypes.train):
    filename = os.path.join(data_dir, TFRECORDS_FILES[type])
    #   ATTENTION:
    #   DO NOT set num_epochs in string_input_producer, otherwise uninitialized variable error occurs...
    #   To resolve this, need to initialize local variables, but it will make it difficult while restoring model checkpoints...
    filename_queue = tf.train.string_input_producer(
        [filename], name='string_input_producer')

    image, height, width, depth, label_id, label_txt, filename = read_and_decode(filename_queue)

    # Collect examples into batch_size batches.
    # (Internally uses a Queue.)
    # We run this in two threads to avoid being a bottleneck.
    images, labels = tf.train.batch(
        [image, height, width, depth, label_id, label_txt, filename],
        batch_size=batch_size,
        num_threads=2,
        capacity=1000 + 3 * batch_size,
        enqueue_many=False,
        allow_smaller_final_batch=True,
        name='input_batching')

    return images, labels


def run():
    data_dir = os.path.join(os.getcwd(), 'drone_data')
    maybe_download_and_convert(data_dir)


if __name__ == '__main__':
    run()

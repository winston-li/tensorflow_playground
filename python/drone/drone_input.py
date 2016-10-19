from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path
from tensorflow.contrib.learn.python.learn.datasets import mnist

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'  # TO CHANGE
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
VALIDATION_DATA_SIZE = 5000
IMAGE_SIZE = 101
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 3

TFRECORDS_FILES = ['train.tfrecords', 'validation.tfrecords', 'test.tfrecords']

# Convert whatever data you have into Standard TensorFlow format (TFRecords). 
# This makes it easier to mix and match data sets and network architectures.


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfr(data_set, to_tfr_path):
    images = data_set.images
    labels = data_set.labels
    num_examples = data_set.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    print('Writing', to_tfr_path)
    writer = tf.python_io.TFRecordWriter(to_tfr_path)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def maybe_download_and_convert(data_dir):
    # Get the data.
    data_sets = mnist.read_data_sets(data_dir, dtype=tf.uint8, reshape=False)
    # Convert to Examples and write the result to TFRecords.
    filenames = [
        os.path.join(data_dir, name + '.tfrecords')
        for name in ['train', 'validation', 'test']
    ]
    for idx, f in enumerate(filenames):
        if not tf.gfile.Exists(f):
            convert_to_tfr(data_sets[idx], f)


# Decode TFRecords
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    return image, tf.one_hot(label, NUM_CLASSES)


class DataTypes:
    train, validation, test = range(3)


def input_pipeline(data_dir, batch_size, type=DataTypes.train):
    filename = os.path.join(data_dir, TFRECORDS_FILES[type])
    #   ATTENTION:
    #   DO NOT set num_epochs in string_input_producer, otherwise uninitialized variable error occurs...
    #   To resolve this, need to initialize local variables, but it will make it difficult while restoring model checkpoints...
    filename_queue = tf.train.string_input_producer(
        [filename], name='string_input_producer')

    image, label = read_and_decode(filename_queue)

    # Collect examples into batch_size batches.
    # (Internally uses a Queue.)
    # We run this in two threads to avoid being a bottleneck.
    images, labels = tf.train.batch(
        [image, label],
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

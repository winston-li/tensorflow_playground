from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(filename, image_raw, label, text, height, width, depth):
    """Build an Example proto for an example.
      Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_raw: string, image raw data 
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'TR'
        height: integer, image height in pixels
        width: integer, image width in pixels
        depth: integer, image depth
      Returns:
        Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/depth': _int64_feature(depth),
        'image/raw': _bytes_feature(image_raw.tostring()),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(text),
        'image/filename':
        _bytes_feature(os.path.join(text, os.path.basename(filename)))
    }))
    return example


def read_and_decode_example(filename_queue, transform=None, one_hot_size=1):
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
    if transform:
        image = transform(image)
    label_id = tf.cast(features['image/class/label'], tf.int32)
    if one_hot_size > 1:
        label_id = tf.one_hot(label_id, one_hot_size)
    label_txt = features['image/class/text']
    filename = features['image/filename']
    return image, height, width, depth, label_id, label_txt, filename

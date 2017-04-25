#############################################################
## SequenceExample [no use of context feature in TFR]
## Note:
##      input feature is a float32 vector per time-step,
##      input feature dimension is the same for all sequences,
##      acutal time-steps may differ for each sequence
#############################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
import tensorflow as tf
import os

def run():
    ### 1: serialize/write part 
    tf.reset_default_graph()

    FEATURE_SIZE_PER_TIMESTEP = 5
    sequences = [[[1.,1.,1.,1.,1.], [2.,3.,4.,5.,6.], [3.,2.,1.,0.,-1.]], 
                 [[4.,3.,1.,2.,5.], [5.,5.,5.,5.,5.], [1.,2.,3.,4.,5.]], 
                 [[1.,0.,0.,0.,1.], [2.,2.,2.,2.,2.]], 
                 [[0.,0.,0.,0.,0.], [2.,1.,0.,-1.,-2.], [4.,8.,12.,16.,20.], [7.,7.,7.,0.,1.]], 
                 [[9.,9.,9.,9.,9.], [8.,8.,1.,1.,1.]], 
                 [[5.,4.,3.,2.,1.], [4.,4.,8.,8.,8.], [3.,3.,3.,6.,6.], [2.,2.,2.,2.,1.], [1.,1.,1.,1.,1.]], 
                 [[3.,0.,3.,0.,3.], [6.,8.,3.,1.,1.], [9.,9.,9.,9.,8.]]]
    label_sequences = [[0, 1, 2], [1, 0, 0], [1, 1], [2, 1, 1, 0], [2, 0], [0, 1, 1, 2, 0], [2, 0, 1]]

    # inputs: A list of input vectors, each input vector is a list of float32 (entries #: FEATURE_SIZE_PER_TIMESTEP)
    # labels: A list of int64
    def make_sequence_example(inputs, labels):
        input_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in inputs]
        label_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[label])) for label in labels]
        feature_list = {
            'inputs': tf.train.FeatureList(feature=input_features),
            'labels': tf.train.FeatureList(feature=label_features)
        }
        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        return tf.train.SequenceExample(feature_lists=feature_lists)

    # Write all examples into a TFRecords file
    data_dir = os.path.join(os.getcwd(), 'sequence_labeling_data')
    tf.gfile.MakeDirs(data_dir)
    output_file = os.path.join(data_dir, 'Sequence_labeling.tfr')
    writer = tf.python_io.TFRecordWriter(output_file)
    for sequence, label_sequence in zip(sequences, label_sequences):
        ex = make_sequence_example(sequence, label_sequence)
        writer.write(ex.SerializeToString())
    writer.close()


    ## 2: deserialize/read part
    tf.reset_default_graph()

    BATCH_SIZE = 4
    FEATURE_SIZE_PER_TIMESTEP = 5

    file_list = [os.path.join(os.getcwd(), 'sequence_labeling_data', 'Sequence_labeling.tfr')]
    print(file_list)
    file_queue = tf.train.string_input_producer(file_list, num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    # Define how to parse the example
    sequence_features = {
        "inputs": tf.FixedLenSequenceFeature([FEATURE_SIZE_PER_TIMESTEP], dtype=tf.float32),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    
    # Parse the example
    _, sequence = tf.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features)
    actual_length = tf.shape(sequence["inputs"])[0]

    # Batch the variable length tensor with dynamic padding
    batch_lengths, batch_sequences, batch_labels = tf.train.batch(
        [actual_length, sequence["inputs"], sequence["labels"]],
        batch_size=BATCH_SIZE,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
        name="input_batching")


    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try: 
            for i in range(2):
                lens, seqs, lbls = sess.run([batch_lengths, batch_sequences, batch_labels])
                print('actual_lengths =', lens)
                print('batch_size=%d, time_steps=%d' % (lbls.shape[0], lbls.shape[1]))
                print('sequences = ', seqs)
                print('labels = ', lbls)      
        except tf.errors.OutOfRangeError as e:
            print('Done')
            print(e.error_code, e.message)
        finally:
            coord.request_stop()


if __name__ == '__main__':
    run()

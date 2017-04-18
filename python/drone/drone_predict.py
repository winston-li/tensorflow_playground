from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os.path
import sys

import drone_input

BATCH_SIZE = 10


def _get_graph_tensors(graph_path, tensors):
    graph_def = tf.GraphDef()
    with open(graph_path, "rb") as f:
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, return_elements=tensors, name='')


def _summarize_predictions(stats, files, preds, labels):
    for i in range(len(files)):
        basename = os.path.basename(files[i]).decode()
        fn_and_id = os.path.splitext(basename)[0]
        filename, id = fn_and_id.split('#')
        stats.setdefault(filename, []).append((id, preds[i], labels[i]))


def _output_predictions(stats, output_dir):
    for fname, id_list in stats.items():
        print('filename = %s' % fname)
        id_list.sort() # ascending frame number
        new_name = os.path.splitext(fname)[0] + '.pdt' 
        with open(os.path.join(output_dir, new_name), 'w') as f:
            f.write('TL_prob, GS_prob, TR_prob\n') # header
            for row in id_list:
                f.write('%.3f, %.3f, %.3f\n' %(row[1][0], row[1][1], row[1][2]))

def _output_error(stats, output_dir):
    for fname, id_list in stats.items():
        # id_list content: (frame_id, [TL,GS,TR prob], [one-hot label])
        error_list = [item for item in id_list if np.argmax(item[1]) != np.argmax(item[2])]
        error_list.sort() # ascending frame number
        new_name = os.path.splitext(fname)[0] + '.err' 
        with open(os.path.join(output_dir, new_name), 'w') as f:
            f.write('zero_based_frame_id, TL_prob, GS_prob, TR_prob, zero_based_true_label(TL,GS,TR)\n') # header
            for row in error_list:
                f.write('%s, %.3f, %.3f, %.3f, %d\n' %(row[0], row[1][0], row[1][1], row[1][2], np.argmax(row[2])))

# Load finalized model graph, and use it to predict new data
def predict(graph_path, data_dir, output_dir):
    tensors = ['Input/images:0', 'Input/dropout_keep_rate:0', 'Model/pred:0']
    images_ph, keep_prob_ph, pred = _get_graph_tensors(graph_path, tensors)

    images, _, _, _, labels, _, filenames = drone_input.input_pipeline(
            data_dir, BATCH_SIZE, drone_input.DataTypes.all, epochs=1)

    stats = {}
    # {'filename': [(frame_id, [TL prob, GS prob, TR prob], [one-hot true_label]), (), ...]}

    with tf.Session() as sess:
        sess.run(tf.initialize_local_variables()) # for string_input_producer in input_pipeline 
        coord = tf.train.Coordinator()
        # Note: QueueRunner created in drone_input.py
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            count = 0
            while not coord.should_stop():
                images_r, labels_r, filenames_r = sess.run([images, labels, filenames])
                data_feed = {
                    images_ph: images_r,
                    keep_prob_ph: 1.0
                }

                predictions = sess.run(
                    pred, feed_dict=data_feed)

                _summarize_predictions(stats, filenames_r, predictions, labels_r)
                count += len(predictions)

                if count % 100 == 0:
                    print('#', end='')
                    sys.stdout.flush()


        except tf.errors.OutOfRangeError:
            print('Done prediction for %d samples' % count)

        finally:
            # When done, ask the threads to stop
            coord.request_stop()

        coord.join(threads)
        _output_predictions(stats, output_dir)
        _output_error(stats, output_dir)


def run():
    tfr_dir = os.path.join(os.getcwd(), 'drone_data', 'tfrecord')
    output_dir = os.path.join(os.getcwd(), 'drone_data')

    model_dir = os.path.join(os.getcwd(), 'models')
    graph_path = os.path.join(model_dir, 'model.pb')
    predict(graph_path, tfr_dir, output_dir)


if __name__ == '__main__':
    run()

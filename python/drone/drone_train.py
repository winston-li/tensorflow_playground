from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input
from builtins import range

import tensorflow as tf
import numpy as np
import time
import os.path
from tqdm import tqdm
import pickle
from datetime import datetime

import drone_model
import drone_input

TRAIN_DATA_SIZE = 24474 + 24913 + 23425  # 72812
# Training Parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 1 #50
DISPLAY_STEP = 10
LOG_STEP = 100
CKPT_STEP = 1000
MAX_TRAINING_STEPS = 2 * TRAIN_DATA_SIZE // BATCH_SIZE


def _get_checkpoint(model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    else:
        return None

# Note: 
#   if use scale_intensity to transform the retrieved tfrecord image,
#   it's suggested to use tf.nn.tanh() instead of tf.nn.relu() as 
#   activation function in model graph.
def scale_intensity(image_tensor):
  # Convert from [0, 255] -> [-1, 1] floats.
  image = tf.cast(image_tensor, tf.float32)
  return tf.cast(image, tf.float32) * (1. / 255) - 1.0


def train(data_dir,
          model_dir,
          log_dir,
          batch_size=BATCH_SIZE,
          max_batches=MAX_TRAINING_STEPS):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')

        images_ph, labels_ph, keep_rate_ph = drone_model.placeholders()
        pred, logits = drone_model.inference(images_ph, keep_rate_ph)
        loss = drone_model.loss(logits, labels_ph)
        avg_loss = drone_model.avg_loss()
        train_op = drone_model.training(loss, LEARNING_RATE, global_step)
        accuracy = drone_model.evaluation(pred, labels_ph)
        images, _, _, _, labels, _, _ = drone_input.input_pipeline(
            data_dir, batch_size, drone_input.DataTypes.train, transform=None)

        merged_summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = _get_checkpoint(model_dir)
        if not ckpt:
            print("Grand New training")
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
        else:
            print("Resume training after %s" % ckpt)
            saver.restore(sess, ckpt)

        coord = tf.train.Coordinator()
        # Start the queue runner, QueueRunner created in drone_input.py
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_writer = tf.train.SummaryWriter(log_dir + '/train', sess.graph)

        acc_step = sess.run(global_step)
        print('accumulated step = %d' % acc_step)
        print('prevous avg_loss = %.3f' % sess.run(avg_loss))
        # Training cycle
        try:
            lst = []
            for step in range(1, MAX_TRAINING_STEPS + 1):
                if coord.should_stop():
                    break

                images_r, labels_r = sess.run([images, labels])
                train_feed = {
                    images_ph: images_r,
                    labels_ph: labels_r,
                    keep_rate_ph: 0.5
                }

                start_time = time.time()
                _, train_loss = sess.run([train_op, loss],
                                         feed_dict=train_feed)
                duration = time.time() - start_time
                lst.append(train_loss)

                assert not np.isnan(
                    train_loss), 'Model diverged with loss = NaN'

                if step % DISPLAY_STEP == 0:
                    examples_per_sec = BATCH_SIZE / duration
                    sec_per_batch = float(duration)
                    print(
                        '%s: step %d, train_loss = %.6f (%.1f examples/sec; %.3f sec/batch)'
                        % (datetime.now(), step, train_loss, examples_per_sec,
                           sec_per_batch))

                if step % LOG_STEP == 0:
                    avg = np.mean(lst)
                    del lst[:]
                    #print('avg loss = %.3f' % avg)
                    sess.run(avg_loss.assign(avg))
                    summary_str = sess.run(merged_summary_op,
                                           feed_dict=train_feed)
                    train_writer.add_summary(summary_str, acc_step + step)

                if step % CKPT_STEP == 0 or step == MAX_TRAINING_STEPS:
                    ckpt_path = os.path.join(model_dir, 'model.ckpt')
                    saver.save(sess, ckpt_path, global_step)

        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs' % (num_epochs))

        finally:
            # When done, ask the threads to stop
            coord.request_stop()

        coord.join(threads)
        train_writer.close()
        sess.close()


def run():
    data_dir = os.path.join(os.getcwd(), 'drone_data', 'tfrecord')
    model_dir = os.path.join(os.getcwd(), 'models')
    log_dir = os.path.join(os.getcwd(), 'logs')
    tf.gfile.MakeDirs(model_dir)
    tf.gfile.MakeDirs(log_dir)

    train(data_dir, model_dir, log_dir, BATCH_SIZE, MAX_TRAINING_STEPS)


if __name__ == '__main__':
    run()

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

import mnist_model
import mnist_input

# Training Parameters
VALIDATION_BATCH_SIZE = mnist_input.VALIDATION_DATA_SIZE
LEARNING_RATE = 0.01
BATCH_SIZE = 10
DISPLAY_STEP = 10
LOG_STEP = 100
CKPT_STEP = 1000
MAX_TRAINING_STEPS = 5500 * 2


def _get_checkpoint(model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    else:
        return None


def train(data_dir,
          model_dir,
          log_dir,
          batch_size=BATCH_SIZE,
          max_batches=MAX_TRAINING_STEPS):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images_ph, labels_ph, keep_rate_ph = mnist_model.placeholders()
        pred, logits = mnist_model.inference(images_ph, keep_rate_ph)
        loss = mnist_model.loss(logits, labels_ph)
        avg_loss = mnist_model.avg_loss()
        train_op = mnist_model.training(loss, LEARNING_RATE, global_step)
        accuracy = mnist_model.evaluation(pred, labels_ph)
        images, labels = mnist_input.input_pipeline(
            data_dir, batch_size, mnist_input.DataTypes.train)
        val_images, val_labels = mnist_input.input_pipeline(
            data_dir, VALIDATION_BATCH_SIZE, mnist_input.DataTypes.validation)

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
        # Start the queue runner, QueueRunner created in mnist_input.py
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_writer = tf.train.SummaryWriter(log_dir + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(log_dir + '/validation',
                                                   sess.graph)

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
                val_images_r, val_labels_r = sess.run([val_images, val_labels])

                train_feed = {
                    images_ph: images_r,
                    labels_ph: labels_r,
                    keep_rate_ph: 0.5
                }
                val_feed = {
                    images_ph: val_images_r,
                    labels_ph: val_labels_r,
                    keep_rate_ph: 1.0
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

                    summary_str, val_loss, val_accuracy = sess.run(
                        [merged_summary_op, loss, accuracy],
                        feed_dict=val_feed)
                    validation_writer.add_summary(summary_str, acc_step + step)

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
        validation_writer.close()
        sess.close()


def run():
    data_dir = os.path.join(os.getcwd(), 'MNIST_data')
    model_dir = os.path.join(os.getcwd(), 'models')
    log_dir = os.path.join(os.getcwd(), 'logs')
    tf.gfile.MakeDirs(model_dir)
    tf.gfile.MakeDirs(log_dir)

    mnist_input.maybe_download_and_convert(data_dir)
    train(data_dir, model_dir, log_dir, BATCH_SIZE, MAX_TRAINING_STEPS)


if __name__ == '__main__':
    run()

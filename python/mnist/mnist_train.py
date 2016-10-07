from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import os.path
from tqdm import tqdm
import pickle
from datetime import datetime

import mnist_model
from tensorflow.examples.tutorials.mnist import input_data

# Training Parameters
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 2
BATCH_SIZE = 10
DISPLAY_STEP = 10
LOG_STEP = 100
CKPT_STEP = 1000


def _is_resume_training(model_path):
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        return True
    else:
        return False


def train(model_path, log_path, train_data, val_data, epochs=TRAINING_EPOCHS):
    if not _is_resume_training(model_path):
        print("Grand New training")
        global_step = 0
        x, y_, keep_rate = mnist_model.get_placeholders()
        pred, logits = mnist_model.inference(x, keep_rate)
        loss = mnist_model.get_loss(logits, y_)
        train_op = mnist_model.training(loss, LEARNING_RATE, global_step)
        accuracy = mnist_model.get_accuracy(pred, y_)
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        train_cost_history, validation_cost_history, validation_accuracy_history = (
            [] for i in range(3))
    else:
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Resume training after ", ckpt.model_checkpoint_path)
            saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +
                                               '.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
            # extract the last part of model_checkpoint_path as global_step
            global_step = int(
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            x = sess.graph.get_tensor_by_name('input_data:0')
            y_ = sess.graph.get_tensor_by_name('label_data:0')
            keep_rate = sess.graph.get_tensor_by_name('dropout_keep_rate:0')
            accuracy = sess.graph.get_tensor_by_name('Accuracy/accuracy:0')
            loss = sess.graph.get_tensor_by_name('Loss/loss:0')
            train_op = sess.graph.get_operation_by_name('Optimizer/train_op')
            pickle_name = 'history.pickle-' + str(global_step)
            history_path = os.path.join(model_path, pickle_name)
            print(history_path)
            with open(history_path, "rb") as f:
                train_cost_history, validation_cost_history, validation_accuracy_history = pickle.load(
                    f)
        else:
            print('Something wrong with specified model path!')
            return

    print("global_step = ", global_step)
    merged_summary_op = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(log_path + '/train', sess.graph)
    validation_writer = tf.train.SummaryWriter(log_path + '/validation',
                                               sess.graph)

    # Training cycle
    for epoch in tqdm(range(TRAINING_EPOCHS), ascii=True):
        epoch_avg_cost = 0.
        total_batch = int(train_data.num_examples / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs, batch_ys = train_data.next_batch(BATCH_SIZE)
            start_time = time.time()
            _, train_cost = sess.run([train_op, loss],
                                     feed_dict={x: batch_xs,
                                                y_: batch_ys,
                                                keep_rate: 0.5})
            duration = time.time() - start_time
            epoch_avg_cost += train_cost / total_batch
            if i % DISPLAY_STEP == 0:
                examples_per_sec = BATCH_SIZE / duration
                sec_per_batch = float(duration)
                print(
                    '%s: step %d, train_loss = %.5f (%.1f examples/sec; %.3f sec/batch)'
                    % (datetime.now(), i, train_cost, examples_per_sec,
                       sec_per_batch))

            cur_step = epoch * total_batch + i + 1
            if i % LOG_STEP == 0:
                summary_str = sess.run(merged_summary_op,
                                       feed_dict={x: batch_xs,
                                                  y_: batch_ys,
                                                  keep_rate: 0.5})
                train_writer.add_summary(summary_str, global_step + cur_step)
                train_cost_history.append(train_cost)
                summary_str, val_cost, val_accuracy = sess.run(
                    [merged_summary_op, loss, accuracy],
                    feed_dict={x: val_data.images,
                               y_: val_data.labels,
                               keep_rate: 1.0})
                validation_writer.add_summary(summary_str,
                                              global_step + cur_step)
                validation_cost_history.append(val_cost)
                validation_accuracy_history.append(val_accuracy)

            if (i + 1) % CKPT_STEP == 0 or (i + 1) == total_batch:
                ckpt_path = os.path.join(model_path, 'model.ckpt')
                saver.save(sess, ckpt_path, global_step=global_step + cur_step)
                pickle_name = 'history.pickle-' + str(global_step + cur_step)
                history_path = os.path.join(model_path, pickle_name)
                with open(history_path, "wb") as f:
                    pickle.dump((train_cost_history, validation_cost_history,
                                 validation_accuracy_history), f)
        # Display average training cost per epoch
        print("Epoch: %d, avg cost= %.9f" % (epoch + 1, epoch_avg_cost))

    train_writer.close()
    validation_writer.close()
    print("Training Finished!")
    sess.close()


def run():
    model_path = os.path.join(os.getcwd(), 'models')
    log_path = os.path.join(os.getcwd(), 'logs')
    tf.gfile.MakeDirs(model_path)
    tf.gfile.MakeDirs(log_path)
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
    train(model_path, log_path, mnist_data.train, mnist_data.validation)


if __name__ == '__main__':
    run()

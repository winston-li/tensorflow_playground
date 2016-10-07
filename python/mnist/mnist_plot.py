from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(train_costs, val_costs, val_accs):
    fig = plt.figure()
    # create figure window
    gs = gridspec.GridSpec(2, 1)
    # Creates grid 'gs' of 2 rows and 1 columns
    ax = plt.subplot(gs[0, 0])
    # Adds subplot 'ax' in grid 'gs' at position [0,0]
    ax.set_ylabel('Cost')
    ax.set_xlabel('Iterations')
    ax.plot(train_costs, 'b-')
    ax.plot(val_costs, 'r-')
    #ax.set_ylim([0, 0.3])
    fig.add_subplot(ax)

    bx = plt.subplot(gs[1, 0])
    bx.set_ylabel('Accuracy')
    bx.plot(val_accs, 'g-')
    #bx.set_ylim([0.94, 1.0])
    fig.add_subplot(bx)
    plt.show()


def plot2(train_costs, val_costs, val_accs):
    plt.figure(1)
    plt.plot(train_costs, 'b-')
    plt.plot(val_costs, 'r-')
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    #pylab.ylim([0,0.3])

    plt.figure(2)
    plt.plot(val_accs, 'g-')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    #pylab.ylim([0.94,1.0])
    plt.show()


def run():
    model_path = os.path.join(os.getcwd(), 'models')
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        latest_num = int(
            ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        print("No history to show")
        return

    pickle_name = 'history.pickle-' + str(latest_num)
    history_path = os.path.join(model_path, pickle_name)
    print(history_path)
    with open(history_path, "rb") as f:
        train_cost_history, validation_cost_history, validation_accuracy_history = pickle.load(
            f)
    plot(train_cost_history, validation_cost_history,
         validation_accuracy_history)
    plot2(train_cost_history, validation_cost_history,
          validation_accuracy_history)


if __name__ == '__main__':
    run()

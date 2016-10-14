from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os.path

from google.protobuf import text_format
from tensorflow.python.framework import graph_util


# copied from tensorflow/python/tools/freeze_graph.py
def _freeze_graph(input_graph, input_saver, input_binary, input_checkpoint,
                  output_node_names, restore_op_name, filename_tensor_name,
                  output_graph, clear_devices):
    """Converts all variables in a graph and checkpoint into constants."""

    if not tf.gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    if input_saver and not tf.gfile.Exists(input_saver):
        print("Input saver file '" + input_saver + "' does not exist!")
        return -1
    if not tf.gfile.Exists(input_checkpoint):
        print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
        return -1
    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    input_graph_def = tf.GraphDef()
    mode = "rb" if input_binary else "r"
    with open(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""
    _ = tf.import_graph_def(input_graph_def, name="")

    with tf.Session() as sess:
        if input_saver:
            with open(input_saver, mode) as f:
                saver_def = tf.train.SaverDef()
                if input_binary:
                    saver_def.ParseFromString(f.read())
                else:
                    text_format.Merge(f.read(), saver_def)
                saver = tf.train.Saver(saver_def=saver_def)
                saver.restore(sess, input_checkpoint)
        else:
            sess.run([restore_op_name],
                     {filename_tensor_name: input_checkpoint})
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names.split(","))

    with tf.gfile.FastGFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


# Restore Model Checkpoint and then Save it as GraphDef ProtoBuf
def _save_graph_def(ckpt_dir, as_text=False):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore from ", ckpt.model_checkpoint_path)
            persistenter = tf.train.import_meta_graph(
                ckpt.model_checkpoint_path + '.meta')
            persistenter.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return None, None
        tf.train.write_graph(
            sess.graph.as_graph_def(),
            ckpt_dir,
            'draft_model.pb',
            as_text=as_text)
        return os.path.join(ckpt_dir,
                            'draft_model.pb'), ckpt.model_checkpoint_path


def finalize_model(model_dir):
    graph_path, ckpt_path = _save_graph_def(model_dir, as_text=True)
    print('Merge GraphDef and Variables from \n %s \n %s' %
          (graph_path, ckpt_path))
    # Load the GraphDef and Freeze it with trained variables altogether
    input_graph_path = graph_path
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = ckpt_path
    output_node_names = "Model/pred"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(model_dir, 'model.pb')
    clear_devices = False
    _freeze_graph(input_graph_path, input_saver_def_path, input_binary,
                  input_checkpoint_path, output_node_names, restore_op_name,
                  filename_tensor_name, output_graph_path, clear_devices)
    print('Finalized Graph %s written' % (output_graph_path))


def run():
    model_dir = os.path.join(os.getcwd(), 'models')
    finalize_model(model_dir)


if __name__ == '__main__':
    run()

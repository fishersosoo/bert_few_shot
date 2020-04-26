# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import common_shapes
from tensorflow.python.ops import standard_ops

tensorboard_dir = r"Z:\bert_few_shot\models\tmp"
if __name__ == '__main__':
    graph = tf.Graph()
    batch_size = 1
    c = 2
    atten_size = 3
    query_size = 4
    unit = 8
    f = 2
    with graph.as_default():
        gt_pos = tf.Variable(np.array([[[2, 3, 4], [3, 4, 1], [-1, 2, 9]],
                                       [[21, 23, 4], [34, 34, 11], [-31, 62, 79]]]), dtype=tf.float32,
                             name="class_vector")

        gt_diplacement = gt_pos -  tf.tile(tf.expand_dims(gt_pos[:, 0, :],axis=[1]),[1,3,1])
        initialize = tf.global_variables_initializer()
        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run([initialize])
            S = sess.run([gt_diplacement])
            print(S)
            writer = tf.summary.FileWriter(tensorboard_dir)
            writer.add_graph(sess.graph)

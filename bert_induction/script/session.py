# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import  common_shapes
from tensorflow.python.ops import standard_ops



tensorboard_dir = r"Z:\bert_few_shot\models\tmp"
if __name__ == '__main__':
    graph = tf.Graph()
    batch_size=3
    sample_size=5
    atten_size=6
    unit=8
    with graph.as_default():
        a = tf.Variable(np.ones([batch_size,sample_size,atten_size]), dtype=tf.float32, name="a")
        a=tf.reshape([batch_size*sample_size,atten_size])
        b = tf.Variable(np.ones([batch_size,atten_size]), dtype=tf.float32, name="kernel")
        rank = a.shape.ndims
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(a, b, [[1], [1]])
        initialize = tf.global_variables_initializer()
        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run([initialize])
            S = sess.run([outputs])
            print(S[0].shape)
            writer = tf.summary.FileWriter(tensorboard_dir)
            writer.add_graph(sess.graph)

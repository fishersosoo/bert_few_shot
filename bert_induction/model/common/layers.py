# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

from model.bert.modeling import dropout, layer_norm


def self_attention(input, attention_size, dropout_prob, name=None):
    """

    Args:
        input: [batch_size, max_seq_len, hidden_size * 2]

    Returns:
         [batch_size, attention_size]
    """
    with tf.variable_scope(name, default_name=name):
        attention_size_hidden_state = tf.layers.dense(input, attention_size, use_bias=False, name="W_a_1",
                                                      activation=tf.tanh)  # [batch_size, max_seq_len, attention_size]
        alpha = tf.nn.softmax(
            tf.squeeze(tf.layers.dense(attention_size_hidden_state, 1, use_bias=False, name="W_a2"), axis=-1),
            axis=-1)  # [batch_size, max_seq_len]
        e = tf.reduce_sum(attention_size_hidden_state * tf.expand_dims(alpha, -1), 1)  # [batch_size, attention_size]
        attention_output = dropout(e, dropout_prob)
        attention_output = layer_norm(attention_output)
        return attention_output


def self_attention_bi_lstm(input, hidden_size, attention_size, dropout_prob,name=None):
    """
    self_attention机制的单层双向lstm
    Args:
        input: float32 tensor with shape [batch_size, max_seq_len, embedding_size]
        hidden_size:
        attention_size:
        dropout_prob:

    Returns:
        [batch_size, attention_size]
    """
    with tf.variable_scope(name,default_name="self_attention_bi_lstm"):
        cell_fw = rnn.BasicLSTMCell(hidden_size)
        cell_bw = rnn.BasicLSTMCell(hidden_size)
        rnn_outputs, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, input, dtype=tf.float32)
        out = tf.concat(rnn_outputs, -1)  # [batch_size, max_seq_len, hidden_size * 2]
        return self_attention(out, attention_size, dropout_prob,name="attention")
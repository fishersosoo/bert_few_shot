# coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.ops import standard_ops, gen_math_ops

from model.common.base_func import get_shape_list, squash


def induction(sample_prediction_vector,
              iter_routing=3,
              ):
    """

    Args:
        sample_prediction_vector: float tensor # [batch_size, k, attention_size]
        iter_routing:

    Returns:
        class vector. float tensor [batch_size, attention_size]
    """
    # tf.logging.info(sample_prediction_vector)
    batch_size, k, attention_size = get_shape_list(sample_prediction_vector, expected_rank=3)
    b = tf.constant(np.zeros([batch_size, 1, k], dtype=np.float32), dtype=tf.float32)
    e = sample_prediction_vector  # [batch_size, k, embedding_size]
    e_hat = squash(tf.layers.dense(e, attention_size, use_bias=True,
                                   name="W_s"))  # [batch_size, k, attention_size]

    e_hat_stopped = tf.stop_gradient(e_hat, name='stop_gradient')
    for r_iter in range(iter_routing):
        with tf.variable_scope("iter_" + str(iter_routing)):
            d = tf.nn.softmax(b, axis=-1)
            if r_iter < iter_routing - 1:  # 前面的迭代，不需要梯度下降
                c_hat_stopped = tf.matmul(d, e_hat_stopped)  # [batch_size, 1, attention_size]
                c = squash(c_hat_stopped)  # [batch_size, 1, attention_size]
                # 更新b
                # b = b + e_hat * c
                c_expend = tf.tile(c, [1, k, 1])  # [batch_size, k, attention_size]
                e_product_c = tf.reduce_sum(tf.multiply(e_hat_stopped, c_expend), axis=-1)  # [batch_size, k]
                e_product_c = tf.expand_dims(e_product_c, -2)  # [batch_size, 1, k]
                b += e_product_c  # [batch_size, 1, k]
            else:  # 最后的迭代，需要梯度下降
                c_hat = tf.matmul(d, e_hat)  # [batch_size, 1, attention_size]
                c = squash(c_hat)  # [batch_size, 1, attention_size]
                c = tf.squeeze(c, [1])  # [batch_size, attention_size]
                # 不需要更新b
    return c


def relation_model(class_vector, support_inputs, learning_rate,
                   query_input, iter_num,
                   h,
                   ):
    """

    Args:
        h:
        class_vector: [batch_size, attention_size]
        query_input: [batch_size, attention_size]

    Returns:
        relation score.  [batch_size, 1]
    """

    batch_size, attention_size = get_shape_list(class_vector, expected_rank=2)
    g = []
    # NTN
    M = tf.get_variable(
        name="M",
        shape=[h, attention_size, attention_size],
        initializer=initializers.get("glorot_uniform"))
    M_bias, support_losses = adjust_relation_weight(class_vector=class_vector,
                                                    support_input=support_inputs,
                                                    M=M,
                                                    learning_rate=learning_rate,
                                                    iter_num=iter_num,
                                                    class_num=1
                                                    )
    tf.summary.scalar("support_losses", support_losses)
    M = M + tf.stop_gradient(M_bias)
    for k in range(h):
        transformed_vector_k = apply_kernel(query_input, M[k])
        g_k = tf.reduce_sum(tf.multiply(class_vector, transformed_vector_k), axis=-1,
                            keepdims=False)  # [batch_size]
        g.append(tf.nn.relu(g_k))
    relation_vector = tf.stack(g, axis=-1)  # [batch_size, h]
    relation_score = cal_relation_score(relation_vector, 1)  # [batch_size, 1]
    return relation_score


def cal_relation_score(relation_vector, class_num):
    """

    Args:
        relation_vector:
        class_num:

    Returns:

    """
    return tf.layers.dense(relation_vector, class_num, activation=tf.nn.sigmoid, name="output", reuse=tf.AUTO_REUSE)


def apply_kernel(input, kernel):
    """

    Args:
        input:
        kernel: [input.shape[-1], units]

    Returns:
        Same shape as input but the last dimension is units
    """
    rank = input.shape.ndims
    if rank > 2:
        outputs = standard_ops.tensordot(input, kernel, [[rank - 1], [0]])
    else:
        outputs = gen_math_ops.mat_mul(input, kernel)
    return outputs


def adjust_relation_weight(class_vector, support_input, M, learning_rate, iter_num, class_num):
    """

    Args:
        class_vector: [batch_size, attention_size]
        support_input: [batch_size, k, hidden_size, ]
        M: [h, attention_size, attention_size]
        learning_rate: float32
        iter_num: int

    Returns:
        M_bias: [h, attention_size, attention_size]

    """
    h, attention_size, attention_size = get_shape_list(M, expected_rank=3)
    batch_size, k, hidden_size = get_shape_list(support_input, expected_rank=3)
    class_vector_extend = tf.expand_dims(class_vector, 1)  # [batch_size, 1, attention_size]
    class_vector_extend = tf.tile(class_vector_extend, [1, k, 1])  # # [batch_size, k, attention_size]
    M_bias = tf.Variable(tf.zeros([h, attention_size, attention_size]), trainable=False, name="M_bias")

    # calculate support loss at least one time
    kernels = M + M_bias
    g = []
    for i in range(h):
        g_k = apply_kernel(support_input, kernels[i])  # [batch_size, k, attention_size]
        g.append(tf.nn.relu(tf.reduce_sum(tf.multiply(class_vector_extend, g_k), axis=-1,
                                          keepdims=False)))  # [batch_size, k]
    relation_vector = tf.stack(g, axis=-1)  # [batch_size, k, h]
    relation_vector = tf.stop_gradient(relation_vector)
    score = cal_relation_score(relation_vector, class_num)  # [batch_size, k, 1]
    support_loss = tf.losses.mean_squared_error(np.ones_like([batch_size, k, 1]), score)

    for iter in range(iter_num):
        kernels = M + M_bias
        g = []
        for i in range(h):
            g_k = apply_kernel(support_input, kernels[i])  # [batch_size, k, attention_size]
            g.append(tf.nn.relu(tf.reduce_sum(tf.multiply(class_vector_extend, g_k), axis=-1,
                                              keepdims=False)))  # [batch_size, k]
        relation_vector = tf.stack(g, axis=-1)  # [batch_size, k, h]
        relation_vector = tf.stop_gradient(relation_vector)
        score = cal_relation_score(relation_vector, class_num)  # [batch_size, k, 1]
        support_loss = tf.losses.mean_squared_error(np.ones_like([batch_size, k, 1]), score)
        grad = tf.gradients(support_loss, M_bias)
        M_bias = tf.assign_sub(M_bias - learning_rate * grad)
    return M_bias, support_loss
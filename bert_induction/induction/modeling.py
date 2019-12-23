# encoding=utf-8
import collections
import copy
import json
import os
import re

import numpy as np
import pandas as pd
import six
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.keras import initializers
from tensorflow.python.ops import standard_ops, gen_math_ops
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

from bert import tokenization
from bert.modeling import layer_norm, dropout
from induction.tokenizer import Tokenizer


class CheckEmbedding:
    def __init__(self, name):
        self.is_equal = True
        self.name = name

    def __call__(self, a, b):
        if self.is_equal:
            if not np.array_equal(a, b):
                tf.logging.warning("{name} not equal!".format(name=self.name))
                self.is_equal = False


def convert_to_ids(text, max_seq_length, tokenizer):
    # tokens=text
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    start_token = ["[CLS]"]
    start_token.extend(tokens)
    tokens = start_token
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    return input_ids, input_mask


class InputText:
    """
    用来存放输入的文本，用于输出展示
    """

    def __init__(self, c, k, query_size):
        self._data = {}
        self.c = c
        self.k = k
        self.query_size = query_size

    def _create_iter_dict(self):
        return {"support_text": {i: [] for i in range(self.c)},
                "query_text": []}

    def write_support_text(self, iter, class_id, text):
        if iter not in self._data:
            self._data[iter] = self._create_iter_dict()
        try:
            self._data[iter]["support_text"][class_id]
        except Exception as e:
            print(self._data)
            raise e
        self._data[iter]["support_text"][class_id].append(text)

    def write_query_text(self, iter, text):
        if iter not in self._data:
            self._data[iter] = self._create_iter_dict()
        self._data[iter]["query_text"].append(text)

    def dump(self, fp):
        with open(fp, "w") as fd:
            json.dump(self._data, fd)


def write_example(fp_in, fp_out, max_seq_length, tokenizer=None, do_predict=False):
    """
    将输入数据转化成TFRecord文件
    每一个 training 迭代写成一个example
    Args:
        do_predict:
        tokenizer:
        max_seq_length:
        fp_in: 输入文件所在目录.
            目录里面包含三个numpy.dump生成的文件 'support_input_text.npy', 'query_input_text.npy', 'query_label.npy'

            support_embedding, string array [training_iter_num, k].
            每个 training_iter 会 reshape 成 [k * max_seq_length * embedding_size] FloatList.

            query_embedding, string array [training_iter_num].
            每个 training_iter 会 reshape 成 [max_seq_length * embedding_size] int64 FloatList.

            query_label，int array [training_iter_num].
            每个 training_iter 会 reshape 成 [1] int64 list.

        fp_out: TFRecord文件路径

    Returns:
        training_iter
    """
    with tf.python_io.TFRecordWriter(fp_out) as writer:
        support_input_text = np.load(os.path.join(fp_in, "support_text.npy"), allow_pickle=True)
        query_input_text = np.load(os.path.join(fp_in, "query_text.npy"), allow_pickle=True)
        if do_predict is False:
            query_label = np.load(os.path.join(fp_in, "query_label.npy"), allow_pickle=True)
        else:
            query_label = None

        training_iter_num, k = support_input_text.shape
        tf.logging.info("support input:{shape} ".format(shape=support_input_text.shape))
        tf.logging.info("query input:{shape} ".format(shape=query_input_text.shape))
        tf.logging.info(str(support_input_text[2].shape))
        for iter_index in range(training_iter_num):
            features = collections.OrderedDict()
            iter_support_embeddings = []
            for text_id, one_text in enumerate(support_input_text[iter_index].reshape(-1)):
                one_text = tokenization.convert_to_unicode(one_text)
                vector = tokenizer.convert_to_vector(one_text, max_len=max_seq_length)
                iter_support_embeddings.append(vector)
            query_vector = tokenizer.convert_to_vector(query_input_text[iter_index], max_len=max_seq_length)

            # 保存时候，所有数据都要flat到 1 维
            features["support_embedding"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=np.array(iter_support_embeddings).reshape(-1)))
            features["query_embedding"] = tf.train.Feature(
                float_list=tf.train.FloatList(value=np.array(query_vector).reshape(-1)))

            if not do_predict:
                features["query_label"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[query_label[iter_index]]))
            else:
                features["query_label"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[0]))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        return training_iter_num


def file_based_input_fn_builder(input_file, config, max_seq_length, is_training, drop_remainder):
    """
    从tf_record 读取样本
    Args:
        max_seq_length:
        input_file:
        config:
        is_training:
        drop_remainder:

    Returns:

    """
    k = config.k
    embedding_size = config.embedding_size

    def _decode_record(record):
        name_to_features = {"support_embedding": tf.FixedLenFeature([k * max_seq_length * embedding_size], tf.float32),
                            "query_embedding": tf.FixedLenFeature([max_seq_length * embedding_size], tf.float32),
                            "query_label": tf.FixedLenFeature([1], tf.int64)}
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.float64:
                t = tf.cast(t, dtype=tf.float32)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]

        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.repeat()

        dataset = dataset.map(lambda record: _decode_record(record), num_parallel_calls=4)
        dataset = dataset.batch(batch_size=batch_size,
                                drop_remainder=drop_remainder)
        dataset = dataset.prefetch(buffer_size=100)

        return dataset

    return input_fn


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def squash(vector, epsilon=1e-9):
    """
    非线性压缩向量
    Args:

        vector: [batch_size, k, vector_len]
        epsilon:

    Returns:
         [batch_size, k, vector_len]
    """
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keepdims=True)  # ||x||^2 [batch_size, k, 1]
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(
        vec_squared_norm + epsilon)  # ||x||^2 / (||x||^2 + 1) / ||x||  [batch_size, k, 1]
    vec_squashed = scalar_factor * vector  # element-wise  [batch_size, k, vector_len]
    return vec_squashed


def self_attention(input, attention_size, dropout_prob):
    """

    Args:
        input: [batch_size, max_seq_len, hidden_size * 2]

    Returns:
         [batch_size, attention_size]
    """
    attention_size_hidden_state = tf.layers.dense(input, attention_size, use_bias=False, name="W_a_1",
                                                  activation=tf.tanh)  # [batch_size, max_seq_len, attention_size]
    alpha = tf.nn.softmax(
        tf.squeeze(tf.layers.dense(attention_size_hidden_state, 1, use_bias=False, name="W_a2"), axis=-1),
        axis=-1)  # [batch_size, max_seq_len]
    e = tf.reduce_sum(attention_size_hidden_state * tf.expand_dims(alpha, -1), 1)  # [batch_size, attention_size]
    attention_output = dropout(e, dropout_prob)
    attention_output = layer_norm(attention_output)
    return attention_output


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


def self_attention_bi_lstm(input, hidden_size, attention_size, dropout_prob, is_training):
    """

    Args:
        input: float32 tensor with shape [batch_size, max_seq_len, embedding_size]
        hidden_size:
        attention_size:
        dropout_prob:
        is_training:

    Returns:
        [batch_size, attention_size]
    """
    if not is_training:
        dropout_prob = 0
    batch_size, max_seq_len, embedding_size = get_shape_list(input, expected_rank=3)

    cell_fw = rnn.BasicLSTMCell(hidden_size)
    cell_bw = rnn.BasicLSTMCell(hidden_size)
    rnn_outputs, _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, input, dtype=tf.float32)
    out = tf.concat(rnn_outputs, -1)  # [batch_size, max_seq_len, hidden_size * 2]
    return self_attention(out, attention_size, dropout_prob)


class InductionModel:
    def __init__(self, config, is_training,
                 support_embeddings, batch_size, seq_len,
                 query_embeddings, query_label, scope=None):
        """

        Args:
            bert_config:
            config:
            is_training:
            support_embeddings: float64 Tensor [batch_size, k * seq_len * embedding_size]
            query_embeddings: float64 Tensor [batch_size, seq_len * embedding_size]
            query_label: int32 Tensor [batch_size]
            scope:
        """
        if not is_training:
            dropout_prob = 0.0
        else:
            dropout_prob = config.dropout_prob
        embedding_size = config.embedding_size
        k = config.k
        support_embeddings = tf.reshape(support_embeddings, [batch_size * k, seq_len, embedding_size])
        query_embeddings = tf.reshape(query_embeddings, [batch_size, seq_len, embedding_size])
        encode_input = tf.concat([support_embeddings, query_embeddings],
                                 0)  # [batch_size * k + 1, seq_len, embedding_size]
        encode_output = self_attention_bi_lstm(encode_input, config.hidden_size, config.attention_size, dropout_prob,
                                               is_training=is_training)
        support_encode = tf.reshape(encode_output[:batch_size * k],
                                    [batch_size, k, config.attention_size])  # [batch_size, k, attention_size]
        query_encode = encode_output[batch_size * k:]  # [batch_size, attention_size]
        self.query_encode = query_encode
        self.support_encode = support_encode
        with tf.variable_scope(scope, default_name="induction"):
            with tf.variable_scope("routing"):
                self.class_vector = induction(support_encode)  # [batch_size, attention_size]

        with tf.variable_scope(scope, default_name="relation"):
            self.relation_score = relation_model(class_vector=self.class_vector, h=config.h,
                                                 query_input=query_encode)  # [batch_size,1]


def create_model(config, is_training, support_embeddings, query_embeddings, batch_size, seq_len,
                 query_label):
    """
    创建模型，model_fn_builder中调用
    Args:
        seq_len:
        config: induction配置
        is_training:
        support_embeddings:
        query_embeddings:
        query_label:

    Returns:
         loss, query_encode, class_vector, support_encode, relation_score
         query_encode, float tensor [attention_size]
         class_vector, float tensor [attention_size]
         support_encode, float tensor [k, attention_size]
         relation_score, float tensor [1]
    """
    model = InductionModel(config=config, seq_len=seq_len,
                           is_training=is_training,
                           support_embeddings=support_embeddings,
                           query_embeddings=query_embeddings,
                           query_label=query_label, batch_size=batch_size)
    with tf.variable_scope("loss"):
        tf.logging.info(query_label)  # [batch_size, 1]
        tf.logging.info(model.relation_score)  # [batch_size, 1]
        loss = tf.losses.mean_squared_error(query_label, model.relation_score)
    return loss, model.query_encode, model.class_vector, model.support_encode, model.relation_score


def model_fn_builder(config, init_checkpoint, learning_rate, batch_size, max_seq_length,
                     num_train_steps, num_warmup_steps, use_tpu, ):
    """

    Args:
        max_seq_length:
        batch_size:
        config:
        init_checkpoint:
        learning_rate:
        num_train_steps:
        num_warmup_steps:
        use_tpu:

    Returns:

    """

    def model_fn(features, labels, mode, params):
        support_embedding = features["support_embedding"]
        query_embedding = features["query_embedding"]
        query_label = features["query_label"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss, query_encode, class_vector, support_encode, relation_score = create_model(seq_len=max_seq_length,
                                                                                        config=config,
                                                                                        is_training=is_training,
                                                                                        support_embeddings=support_embedding,
                                                                                        query_embeddings=query_embedding,
                                                                                        batch_size=batch_size,
                                                                                        query_label=query_label)

        # init_checkpoint
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        # logging checkpoint
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(loss, query_label, predict_class):
                accuracy = tf.metrics.accuracy(
                    labels=query_label, predictions=predict_class)
                loss = loss
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [loss, query_label, relation_score])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)

        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "query_embedding": query_encode, "class_vector": class_vector,
                    "support_embedding": support_encode, "relation_score": relation_score},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


class ModelConfig:
    # 模型超参数（需要config文件中配置，和导出的模型一起打包）
    # c: 类别个数
    # k: 每类样本数量
    # seq_len: 每个样本特征数量
    # query_size: query set 样本数量

    def __init__(self, k=5, dropout_prob=.1, embedding_size=300, hidden_size=128, attention_size=64, h=100):
        """
        模型超参数（需要config文件中配置，和导出的模型一起打包）
        Args:

        """
        self.k = k
        self.dropout_prob = dropout_prob
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.h = h

    @classmethod
    def from_dict(cls, json_obj):
        """

        Args:
            json_obj:

        Returns:

        """
        config = cls()
        for (k, v) in six.iteritems(json_obj):
            config.__dict__[k] = v
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """

        Args:
            json_file: json文件路径

        Returns:

        """
        with tf.gfile.GFile(json_file) as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def classifier(embedding_file, dict_path,
               data_dir,
               config_file,
               output_dir,
               max_seq_length=32,
               init_checkpoint=None,
               do_train=False,
               do_eval=False,
               do_predict=False,
               train_batch_size=32,
               eval_batch_size=8,
               predict_batch_size=8,
               num_train_epochs=3.0,
               warmup_proportion=0.1,
               save_checkpoints_steps=1000,
               iterations_per_loop=1000,
               learning_rate=5e-5,
               use_tpu=False, tpu_name=None,
               tpu_zone=None, gcp_project=None,
               master=None, num_tpu_cores=8,
               predict_class_num=5,
               use_cpu=True
               ):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not do_train and not do_eval and not do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    model_config = ModelConfig.from_json_file(config_file)
    tf.gfile.MakeDirs(output_dir)
    tokenizer = Tokenizer(path=embedding_file, dict_path=dict_path)
    tpu_cluster_resolver = None
    if use_tpu and tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=tpu_zone, project=gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=master,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))
    # if use_cpu:
    #     run_config = run_config.replace(session_config=tf.ConfigProto(log_device_placement=True,
    #                                                                   device_count={'GPU': 1}))
    run_config = run_config.replace(
        session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    example_nums = None
    num_train_steps = None
    num_warmup_steps = None
    batch_size = 1
    if do_train:
        batch_size = train_batch_size
    if do_predict:
        batch_size = predict_batch_size
    if do_eval:
        batch_size = eval_batch_size

    if do_train:
        example_nums = write_example(os.path.join(data_dir, "train"),
                                     os.path.join(data_dir, "train.tf_record"),
                                     max_seq_length=max_seq_length,
                                     tokenizer=tokenizer)
        num_train_steps = int(
            example_nums / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

    model_fn = model_fn_builder(config=model_config, init_checkpoint=init_checkpoint, max_seq_length=max_seq_length,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps,
                                use_tpu=use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    if do_train:
        train_file = os.path.join(data_dir, "train.tf_record")
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", example_nums)
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(train_file, model_config, max_seq_length, True, True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if do_eval:
        eval_steps = None
        eval_file = os.path.join(data_dir, "eval.tf_record")
        eval_examples_num = write_example(os.path.join(data_dir, "eval"),
                                          eval_file,
                                          max_seq_length=max_seq_length,
                                          tokenizer=tokenizer)
        if use_tpu:
            assert eval_examples_num % eval_batch_size == 0
            eval_steps = int(eval_examples_num // eval_batch_size)
        eval_drop_remainder = True if use_tpu else False
        eval_input_fn = file_based_input_fn_builder(eval_file, model_config, max_seq_length, False, eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        # 输出结果
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if do_predict:
        predict_file = os.path.join(data_dir, "predict.tf_record")
        predict_run_num = write_example(os.path.join(data_dir, "predict"),
                                        predict_file, do_predict=True,
                                        max_seq_length=max_seq_length,
                                        tokenizer=tokenizer)
        predict_examples_num = int(predict_run_num / predict_class_num)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d ", predict_examples_num)
        tf.logging.info("  Num runs = %d ", predict_run_num)
        tf.logging.info("  Batch size = %d", predict_batch_size)
        predict_drop_remainder = True if use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            config=model_config,
            max_seq_length=max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder
        )
        result = [one for one in estimator.predict(input_fn=predict_input_fn)]
        tf.logging.info("result size = %d", len(result))
        # checker
        check_query_embedding_equal = CheckEmbedding("query embeddings")
        check_support_embedding_equal = CheckEmbedding("support embeddings")
        check_class_vector_equal = CheckEmbedding("class vector")
        # file path
        output_predict_file = os.path.join(output_dir, "test_results.csv")
        output_embeddings_file = os.path.join(output_dir, "test_embeddings.csv")
        output_class_vector_fp = os.path.join(output_dir, "class_vector.csv")
        output_support_embeddings_fp = os.path.join(output_dir, "support_embeddings.csv")
        # data
        support_embeddings = {"class_id": [], "sample_id": [], "embeddings": []}
        class_vectors = {"class_id": [], "embeddings": []}
        # get class vector for debug
        for class_id in range(predict_class_num):
            support_embedding = None
            class_vector = None
            for run_id in range(class_id, predict_run_num, predict_class_num):
                if support_embedding is None:
                    support_embedding = result[run_id]["support_embedding"]
                else:
                    check_support_embedding_equal(support_embedding, result[run_id]["support_embedding"])
                if class_vector is None:
                    class_vector = result[run_id]["class_vector"]
                else:
                    check_class_vector_equal(class_vector, result[run_id]["class_vector"])

            for sample_id, embedding in enumerate(support_embedding):
                support_embeddings["class_id"].append(class_id)
                support_embeddings["sample_id"].append(sample_id)
                support_embeddings["embeddings"].append(embedding.tolist())
            class_vectors["embeddings"].append(class_vector)
            class_vectors["class_id"].append(class_id)

        # write file
        pd.DataFrame(class_vectors).to_csv(output_class_vector_fp, index=False)
        pd.DataFrame(support_embeddings).to_csv(output_support_embeddings_fp, index=False)

        # get query sample result
        result_data = {"sample_id": [], "prediction": [], "embeddings": []}
        for class_index in range(predict_class_num):
            result_data[str(class_index)] = []
        for sample_id in range(predict_examples_num):
            result_data["sample_id"].append(sample_id)
            probabilities = []
            query_embedding = None
            for class_id in range(predict_class_num):
                result_id = sample_id * predict_class_num + class_id
                class_probability = result[result_id]["relation_score"][0]
                probabilities.append(class_probability)
                result_data[str(class_id)].append(class_probability)
                if query_embedding is None:
                    query_embedding = result[result_id]["query_embedding"]
                else:
                    check_query_embedding_equal(result[result_id]["query_embedding"], query_embedding)
            result_data["embeddings"].append(query_embedding.tolist())
            result_data["prediction"].append(np.argmax(probabilities))
        result_df = pd.DataFrame(result_data)
        result_df[["sample_id", "embeddings"]].to_csv(output_embeddings_file, index=False)
        result_df.drop(columns=["embeddings"]).to_csv(output_predict_file, index=False)


def fine_tune():
    project_path = "/home/bert_few_shot"
    model_path = os.path.join(project_path, "models")
    model_config_fp = os.path.join(model_path, "test", "model_config.json")
    with open(model_config_fp, 'w') as model_config_fd:
        json.dump(ModelConfig(k=5,
                              dropout_prob=0.1,
                              hidden_size=128,
                              attention_size=64,
                              h=100).to_dict(),
                  model_config_fd)
    classifier(embedding_file=os.path.join(model_path, "vector", "merge_sgns_bigram_char300.txt"),
               dict_path=os.path.join(model_path, "vector", "user_dict.txt"),
               data_dir=os.path.join(project_path, "data", "output"),
               config_file=model_config_fp,
               init_checkpoint=None,
               max_seq_length=32,
               train_batch_size=32,
               eval_batch_size=1,
               predict_batch_size=1,
               num_train_epochs=32.0,
               output_dir=os.path.join(model_path, "test"),
               do_train=True,
               iterations_per_loop=100,
               learning_rate=5e-4,
               save_checkpoints_steps=10000,
               use_cpu=False
               )


def predict(output_dir):
    project_path = "/home/bert_few_shot"
    model_path = os.path.join(project_path, "models")
    model_config_fp = os.path.join(model_path, "test", "model_config.json")
    classifier(embedding_file=os.path.join(model_path, "vector", "merge_sgns_bigram_char300.txt"),
               dict_path=os.path.join(model_path, "vector", "user_dict.txt"),
               data_dir=os.path.join(project_path, "data", "output", "self_support"), predict_class_num=1,
               config_file=model_config_fp,
               init_checkpoint=os.path.join(model_path, "test", "model.ckpt-10000"),
               max_seq_length=64,
               predict_batch_size=1,
               output_dir=output_dir,
               do_predict=True,
               use_cpu=False
               )


if __name__ == "__main__":
    import logging

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fine_tune()
    predict("/home/bert_few_shot/data/output/self_support/result")

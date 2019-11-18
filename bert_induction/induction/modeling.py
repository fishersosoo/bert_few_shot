# encoding=utf-8
import collections
import copy
import os
import re
import json
import numpy as np
import six
import tensorflow as tf

from bert import tokenization
from bert.modeling import BertModel, BertConfig
import pandas as pd


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


class InputText():
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
        tokenizer:
        max_seq_length:
        fp_in: 输入文件所在目录.
            目录里面包含三个numpy.dump生成的文件 'support_input_text', 'query_input_text', 'query_label'

            support_input_text, string array [training_iter, c, k].
            每个 training_iter 会 reshape 成 [c * k * max_seq_length] int64 list.

            query_input_text, string array [training_iter, query_size].
            每个 training_iter 会 reshape 成 [query_size * max_seq_length] int64 list.

            query_label，int array [training_iter, query_size].
            需要转化成one-hot编码 int array [training_iter, query_size, c].
            每个 training_iter 会 reshape 成 [query_size * c] int64 list.

        fp_out: TFRecord文件路径

    Returns:
        training_iter
    """
    with tf.python_io.TFRecordWriter(fp_out) as writer:
        # if tokenizer is None:
        #     tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        support_input_text = np.load(os.path.join(fp_in, "support_text.npy"), allow_pickle=True)
        query_input_text = np.load(os.path.join(fp_in, "query_text.npy"), allow_pickle=True)
        if do_predict is False:
            query_label = np.load(os.path.join(fp_in, "query_label.npy"), allow_pickle=True)
        else:
            query_label = None

        training_iter, c, k = support_input_text.shape
        _training_iter, query_size = query_input_text.shape
        tf.logging.info("support input:{shape} ".format(shape=support_input_text.shape))
        tf.logging.info("query input:{shape} ".format(shape=query_input_text.shape))
        input_text = InputText(c, k, query_size)
        tf.logging.info(str(support_input_text[2].shape))
        for iter_index in range(training_iter):
            features = collections.OrderedDict()
            iter_support_input_ids = []
            iter_support_input_mask = []
            iter_query_input_ids = []
            iter_query_input_mask = []
            for text_id, one_text in enumerate(support_input_text[iter_index].reshape(-1)):
                one_text = tokenization.convert_to_unicode(one_text)
                ids, mask = convert_to_ids(one_text, max_seq_length, tokenizer)
                if iter_index == 0 or iter_index == training_iter - 1:
                    tf.logging.info("support text {text_id}:{text}\n ids:{ids}\n mask:{mask}".format(
                        text_id=text_id,
                        text=one_text,
                        mask=mask,
                        ids=ids
                    ))
                iter_support_input_ids.append(ids)
                iter_support_input_mask.append(mask)
                input_text.write_support_text(iter_index, int(text_id / k), one_text)
            for text_id,one_query_text in enumerate(query_input_text[iter_index]):
                input_text.write_query_text(iter_index, one_query_text)
                one_query_text = tokenization.convert_to_unicode(one_query_text)
                query_ids, mask = convert_to_ids(one_query_text, max_seq_length, tokenizer)
                if iter_index == 0 or iter_index == training_iter - 1:
                    tf.logging.info("query text:{text}\n ids:{ids}\n mask:{mask}".format(
                        text=one_query_text,
                        ids=query_ids,
                        mask=mask
                    ))
                    if query_label is not None:
                        tf.logging.info("label:{label}".format(label=query_label[iter_index][text_id]))
                iter_query_input_ids.append(query_ids)
                iter_query_input_mask.append(mask)
            # 保存时候，所有数据都要flat到 1 维
            features["support_input_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array(iter_support_input_ids).reshape(-1)))
            features["support_input_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array(iter_support_input_mask).reshape(-1)))
            features["query_input_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array(iter_query_input_ids).reshape(-1)))
            features["query_input_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=np.array(iter_query_input_mask).reshape(-1)))
            if query_label is not None:
                features["query_label"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=query_label[iter_index].reshape(-1)))
            else:
                features["query_label"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[0] * (query_size * c)))
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        input_text.dump(os.path.join(fp_in, "text.json"))
        return training_iter


def file_based_input_fn_builder(input_file, config, max_seq_length, is_training, drop_remainder):
    """

    Args:
        input_file:
        config:
        is_training:
        drop_remainder:

    Returns:

    """
    c = config.c
    k = config.k
    query_size = config.query_size

    def _decode_record(record):
        name_to_features = {"support_input_ids": tf.FixedLenFeature([c * k * max_seq_length], tf.int64),
                            "support_input_mask": tf.FixedLenFeature([c * k * max_seq_length], tf.int64),
                            "query_input_ids": tf.FixedLenFeature([query_size * max_seq_length], tf.int64),
                            "query_input_mask": tf.FixedLenFeature([query_size * max_seq_length], tf.int64),
                            "query_label": tf.FixedLenFeature([query_size * c], tf.int64)}
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        # for name in list(example.keys()):
        #     t = example[name]
        #     if t.dtype == tf.int64:
        #         t = tf.to_int32(t)
        #     example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]

        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.repeat()


        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
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


def transformation(sample_vector,
                   transformation_matrix
                   ):
    """

    Args:
        sample_vector: float Tensor of shape [batch_size, c, k, seq_length].
        transformation_matrix: float Tensor of shape [seq_length, seq_length].
        initializer_range: float. transformation_matrix initialization range.
        transformation_matrix_name:  Name of the transformation matrix

    Returns:
        float Tensor of shape [batch_size, c, k, seq_length].

    """
    batch_size, c, k, seq_length = get_shape_list(sample_vector, expected_rank=4)
    transformation_matrix_expended = tf.expand_dims(transformation_matrix, 0)  # [1, seq_length, seq_length]
    transformation_matrix_expended = tf.expand_dims(transformation_matrix_expended, 0)  # [1, 1, seq_length, seq_length]
    transformation_matrix_expended = tf.expand_dims(transformation_matrix_expended,
                                                    0)  # [1, 1, 1, seq_length, seq_length]
    transformation_matrix_expended = tf.tile(transformation_matrix_expended,
                                             [batch_size, c, k, 1, 1])  # [batch_size, c, k, seq_length, seq_length]
    sample_prediction_vector = tf.matmul(transformation_matrix_expended, tf.expand_dims(sample_vector, -1))
    # [batch_size, c, k, seq_length, 1] = [batch_size, c, k, seq_length, seq_length] * [batch_size, c, k, seq_length.1]
    return tf.squeeze(sample_prediction_vector, axis=[-1])


def squash(vector, epsilon=1e-9):
    """

    Args:
        vector: [batch_size, c, 1, seq_len]
        epsilon:

    Returns:

    """
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keepdims=True)  # [batch_size, c, 1, 1]
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(
        vec_squared_norm + epsilon)  # [batch_size, c, 1, 1]
    vec_squashed = scalar_factor * vector  # element-wise [batch_size, c, 1, seq_len]
    return vec_squashed

def induction_reduce_sum(sample_prediction_vector):
    """
    简单地将样本向量相加作为
    Args:
        sample_prediction_vector:

    Returns:
        class vector. float tensor [batch_size, c,  class_vector_len]

    """
    pass
def induction(sample_prediction_vector,
              class_vector_len,
              iter_routing=3,
              ):
    """

    Args:
        class_vector_len: scalar 表示类向量的维度
        sample_prediction_vector: float tensor [batch_size, c, k, seq_length]
        iter_routing:
        initializer_range:

    Returns:
        class vector. float tensor [batch_size, c,  class_vector_len]
    """
    # tf.logging.info(sample_prediction_vector)
    batch_size, c, k, seq_length = get_shape_list(sample_prediction_vector, expected_rank=4)
    b = tf.constant(np.zeros([batch_size, c, 1, k], dtype=np.float32), dtype=tf.float32)
    e = sample_prediction_vector  # [batch_size, c, k, seq_len]
    e_hat = tf.layers.dense(e, seq_length)  # [batch_size, c, k, seq_len]

    e_hat_stopped = tf.stop_gradient(e_hat, name='stop_gradient')
    # e_hat_stopped=e_hat
    for r_iter in range(iter_routing):
        with tf.variable_scope("iter_" + str(iter_routing)):
            d = tf.nn.softmax(b, axis=-1)

            if r_iter < iter_routing - 1:  # 前面的迭代，不需要梯度下降
                c_hat_stopped = tf.matmul(d, e_hat_stopped)  # [batch_size, c, 1, seq_len]
                c = squash(c_hat_stopped)  # [batch_size, c, 1, seq_len]
                # 更新b
                # b=b + e_hat * c
                c_expend = tf.tile(c, [1, 1, k, 1])  # [batch_size, c, k, seq_len]
                e_product_c = tf.reduce_sum(tf.multiply(e_hat_stopped, c_expend), axis=-1)  # [batch_size, c, k]
                e_product_c = tf.expand_dims(e_product_c, -2)  # [batch_size, c, 1, k]
                b += e_product_c  # [batch_size, c, 1, k]
            if r_iter == iter_routing - 1:  # 最后的迭代，需要梯度下降
                c_hat = tf.matmul(d, e_hat)  # [batch_size, c, 1, seq_len]
                c = squash(c_hat)  # [batch_size, c, 1, seq_len]
                c = tf.squeeze(c, [-2])  # [batch_size, c,  seq_len]
                # 不需要更新b
                # c_hat = tf.squeeze(c_hat, [-2])  # [batch_size, c, seq_len]
                # c = squash(c_hat)  # [batch_size, c, seq_len]
    return c


def relation_model(class_vector,
                   query_input,
                   h=2,
                   initializer_range=0.02
                   ):
    """

    Args:
        class_vector: [batch_size, c, seq_len]
        query_input: [batch_size, query_size, seq_len]
        initializer_range:

    Returns:
        relation score. [batch_size, query_size, c]
    """
    batch_size, c, seq_len = get_shape_list(class_vector)
    batch_size, query_size, seq_len = get_shape_list(query_input)

    transformed_vector = []

    for k in range(h):
        transformed_vector_k = tf.layers.dense(query_input, seq_len, use_bias=False,
                                               name="M_{k}".format(k=k))  # [batch_size, query_size, seq_len]
        transformed_vector.append(transformed_vector_k)
    stacked=tf.stack(transformed_vector, axis=2)  # [batch_size, query_size, h, seq_len]
    transformed_vector = tf.expand_dims(stacked,
                                        2)  # [batch_size, query_size, 1, h, seq_len]
    transformed_vector = tf.tile(transformed_vector, [1, 1, c, 1, 1])  # [batch_size, query_size, c, h, seq_len]

    # class_vector: [batch_size, c, seq_len] -> [batch_size, query_size, c, h, seq_len]
    class_vector_extend=tf.expand_dims(class_vector, 1) # [batch_size, 1, c, seq_len]
    class_vector_extend = tf.expand_dims(class_vector_extend, -2) # [batch_size, 1, c, 1, seq_len]
    class_vector_extend = tf.tile(class_vector_extend, [1, query_size, 1, h, 1])

    g = tf.reduce_sum(tf.multiply(class_vector_extend, transformed_vector), axis=-1,
                      keepdims=False)  # [batch_size, query_size, c, h]
    v = tf.nn.sigmoid(g)
    relation_score = tf.layers.dense(v, 1, activation=tf.nn.sigmoid)  # [batch_size, query_size, c, 1]
    final_score = tf.squeeze(relation_score,[-1]) # [batch_size, query_size, c]
    return final_score


class InductionModel():
    def __init__(self, bert_config, config, is_training,
                 support_input_ids, support_input_mask, batch_size,
                 query_input_ids, query_input_mask, query_label, scope=None):
        """

        Args:
            bert_config:
            config:
            is_training:
            support_input_ids: int32 Tensor [batch_size * c * k, seq_len]
            support_input_mask: int32 Tensor [batch_size * c * k, seq_len]
            query_input_ids: int32 Tensor [batch_size * query_size, seq_len]
            query_input_mask: int32 Tensor [batch_size * query_size, seq_len]
            query_label: int32 Tensor [batch_size * query_size, c] 经过one-hot编码
            scope:
        """
        c = config.c
        k = config.k
        query_size = config.query_size
        support_input_shape = get_shape_list(support_input_ids, expected_rank=2)
        # batch_size = support_input_shape[0]
        tf.logging.info(support_input_ids)

        max_seq_len = int(support_input_shape[1] / (c * k))
        config = copy.deepcopy(config)
        # reshape input
        support_input_ids = tf.reshape(support_input_ids, [batch_size * c * k, max_seq_len])
        support_input_mask = tf.reshape(support_input_mask, [batch_size * c * k, max_seq_len])
        query_input_ids = tf.reshape(query_input_ids, [batch_size * query_size, max_seq_len])
        query_input_mask = tf.reshape(query_input_mask, [batch_size * query_size, max_seq_len])
        input_ids = tf.concat([support_input_ids, query_input_ids], axis=0)
        input_mask = tf.concat([support_input_mask, query_input_mask], axis=0)
        bert_model = BertModel(bert_config, is_training=is_training, input_ids=input_ids, input_mask=input_mask,
                               token_type_ids=None, use_one_hot_embeddings=False)
        output_layer = bert_model.get_pooled_output()
        embedding_size = output_layer.shape[-1].value
        support_input, query_input = tf.split(output_layer, [batch_size * c * k, batch_size * query_size])

        support_input = tf.reshape(support_input, [batch_size, c, k, embedding_size])

        query_input = tf.reshape(query_input, [batch_size, query_size, embedding_size])
        self.output_layer = query_input

        query_label = tf.reshape(query_label, [batch_size, query_size, c])
        if not is_training:
            # dropout_prob设为0
            pass
        # batch_size, c, k, seq_length = get_shape_list(support_input, expected_rank=4)
        # _, query_size, _ = get_shape_list(support_input, expected_rank=3)
        with tf.variable_scope(scope, default_name="induction"):
            with tf.variable_scope("routing"):
                self.class_vector = induction(support_input, 768)
        with tf.variable_scope(scope, default_name="relation"):
            self.relation_score = relation_model(class_vector=self.class_vector,
                                                 query_input=query_input)  # [batch_size, query_size, c]
        with tf.variable_scope(scope, default_name="loss"):
            self.loss = tf.losses.mean_squared_error(query_label, self.relation_score)


def create_model(bert_config, config, is_training, support_input_ids, support_input_mask, query_input_ids, batch_size,
                 query_input_mask, query_label):
    """
    创建模型，model_fn_builder中调用
    Args:
        bert_config: bert配置
        config: induction配置
        is_training:
        support_input_ids:
        support_input_mask:
        query_input_ids:
        query_input_mask:
        query_label:

    Returns:

    """
    model = InductionModel(bert_config=bert_config,
                           config=config,
                           is_training=is_training,
                           support_input_ids=support_input_ids,
                           support_input_mask=support_input_mask,
                           query_input_ids=query_input_ids,
                           query_input_mask=query_input_mask,
                           query_label=query_label, batch_size=batch_size)
    with tf.variable_scope("loss"):
        query_label = tf.reshape(query_label, [get_shape_list(query_label)[0], config.query_size, config.c])
        loss = tf.losses.mean_squared_error(query_label, model.relation_score)
    return loss, model.relation_score, model.output_layer


def model_fn_builder(bert_config, config, init_checkpoint, learning_rate, batch_size,
                     num_train_steps, num_warmup_steps, use_tpu, ):
    """

    Args:
        config:
        init_checkpoint:
        learning_rate:
        num_train_steps:
        num_warmup_steps:
        use_tpu:

    Returns:

    """

    def model_fn(features, labels, mode, params):
        support_input_ids = features["support_input_ids"]
        support_input_mask = features["support_input_mask"]
        query_input_ids = features["query_input_ids"]
        query_input_mask = features["query_input_mask"]
        query_label = features["query_label"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        loss, relation_score, output_layer = create_model(bert_config=bert_config,
                                                          config=config,
                                                          is_training=is_training,
                                                          support_input_ids=support_input_ids,
                                                          support_input_mask=support_input_mask,
                                                          query_input_ids=query_input_ids,
                                                          query_input_mask=query_input_mask, batch_size=batch_size,
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

            def metric_fn(loss, query_label, relation_score):
                predictions = tf.argmax(relation_score, axis=-1, output_type=tf.int32)
                label_ids = tf.argmax(query_label, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions)
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
                predictions={"relation_score": relation_score, "output_layer": output_layer},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


class ModelConfig():
    # 模型超参数（需要config文件中配置，和导出的模型一起打包）
    # c: 类别个数
    # k: 每类样本数量
    # seq_len: 每个样本特征数量
    # query_size: query set 样本数量

    def __init__(self, c=2, k=5, seq_len=768, query_size=10):
        """
        模型超参数（需要config文件中配置，和导出的模型一起打包）
        Args:
            c: 类别个数
            k: 每类样本数量
            seq_len: 每个样本特征数量
            query_size: query set 样本数量
        """
        self.c = c
        self.k = k
        self.seq_len = seq_len
        self.query_size = query_size

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


def classifier(vocab_file,
               data_dir,
               config_file,
               bert_config_file,
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
               use_cpu=True
               ):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not do_train and not do_eval and not do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    bert_config = BertConfig.from_json_file(bert_config_file)
    model_config = ModelConfig.from_json_file(config_file)
    tf.gfile.MakeDirs(output_dir)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
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
        session_config=tf.ConfigProto( gpu_options=tf.GPUOptions(allow_growth=True)))

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

    model_fn = model_fn_builder(bert_config=bert_config, config=model_config, init_checkpoint=init_checkpoint,
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
        predict_examples_num = write_example(os.path.join(data_dir, "predict"),
                                             predict_file, do_predict=True,
                                             max_seq_length=max_seq_length,
                                             tokenizer=tokenizer)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d ",
                        predict_examples_num)
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
        output_predict_file = os.path.join(output_dir, "test_results.csv")
        output_embeddings_file = os.path.join(output_dir, "test_embeddings.csv")
        result_data = {"sample_id": [], "score": [], "predict_class": [], "embeddings": []}
        predict_class_num = 5
        run_num = int(np.ceil(predict_class_num / model_config.c))  # 一个样本需要跑多少次才能得到所有类的分数
        sample_num = int(len(result) / run_num * model_config.query_size)
        embedding_not_equal = False
        tf.logging.info('relation_score shape: {shape}'.format(shape=result[0]["relation_score"].shape))
        for sample_id in range(sample_num):
            run_id_start = int(sample_id / model_config.query_size) * run_num
            sample_index_in_run = sample_id % model_config.query_size
            embedding = None
            score = []
            for run_index in range(run_num):
                run_id = run_id_start + run_index
                if embedding is None:
                    embedding = result[run_id]["output_layer"][sample_index_in_run]
                else:
                    if not embedding_not_equal:
                        if not np.array_equal(embedding, result[run_id]["output_layer"][sample_index_in_run]):
                            tf.logging.warning("embedding not equal!!!!!!!")
                            tf.logging.warning(embedding)
                            tf.logging.warning(result[run_id]["output_layer"][sample_index_in_run])
                            embedding_not_equal = True
                score.extend(result[run_id]["relation_score"][sample_index_in_run])
            result_data["sample_id"].append(sample_id)
            result_data["score"].append(score)
            result_data["predict_class"].append(np.argmax(score))
            result_data["embeddings"].append(embedding.tolist())
        result_df = pd.DataFrame(result_data)
        result_df[["sample_id", "score", "predict_class"]].to_csv(output_predict_file, index=False)
        result_df[["sample_id", "embeddings"]].to_csv(output_embeddings_file, index=False)


if __name__ == "__main__":
    import logging

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # tokenizer = tokenization.FullTokenizer(vocab_file=r"Z:\bert_few_shot\models\chinese_L-12_H-768_A-12\vocab.txt")
    # print(write_example(fp_in=r"Z:\bert_few_shot\data\output\train",
    #                     fp_out=r"Z:\bert_few_shot\data\output\train\train.tf_record",
    #                     tokenizer=tokenizer,
    #                     max_seq_length=32))
    project_path = "/home/bert_few_shot"
    pre_train_model = "chinese_L-12_H-768_A-12"
    model_path = os.path.join(project_path, "models")
    model_config_fp = os.path.join(model_path, "test", "model_config.json")
    with open(model_config_fp, 'w') as model_config_fd:
        json.dump(ModelConfig(c=2, k=5, query_size=32).to_dict(), model_config_fd)
    classifier(vocab_file=os.path.join(model_path, pre_train_model, "vocab.txt"),
               data_dir=os.path.join(project_path, "data", "output"),
               config_file=model_config_fp,
               bert_config_file=os.path.join(model_path, pre_train_model, "bert_config.json"),
               init_checkpoint=os.path.join(model_path, pre_train_model, "bert_model.ckpt"),
               # init_checkpoint=os.path.join(model_path, "test", "model.ckpt-250"),
            max_seq_length=64,
               train_batch_size=2,
               eval_batch_size=1,
               predict_batch_size=1,
               num_train_epochs=4.0,
               output_dir=os.path.join(model_path, "test"),
               do_train=True,
               # do_predict=True,
               iterations_per_loop=100,
               learning_rate=5e-4,
               save_checkpoints_steps=10000,
               use_cpu=False
               )

# encoding=utf-8
import collections
import json
import os

import numpy as np
import tensorflow as tf

from model.albert_zh import modeling
from model.albert_induction.config import ModelConfig
from model.albert_induction.layers import induction_with_DR, relation_model
from model.albert_zh.bert_utils import get_shape_list
from model.albert_zh.modeling import get_assignment_map_from_checkpoint
from model.albert_zh.optimization import create_optimizer
from model.common.base_func import log_variables


class InductionModel:
    def __init__(self,
                 induction_config,
                 is_training,
                 support_text_ids,
                 support_text_mask,
                 seq_len,
                 query_text_ids,
                 query_text_mask,
                 query_label,
                 bert_config,
                 scope=None):
        """
        create a few-shot classification model
        Args:
            induction_config (ModelConfig):
            support_text_ids: int64 Tensor [batch_size, c * k * seq_len]
            support_text_mask: same shape as support_text_ids
            query_text_ids: int64 Tensor [batch_size, query_size * seq_len]
            query_text_mask: same shape as query_text_ids
            query_label: int64 Tensor [batch_size, query_size * c]
            is_training:
            scope:
        """
        if not is_training:
            dropout_prob = 0.0
        else:
            dropout_prob = induction_config.dropout_prob
        c = induction_config.c
        k = induction_config.k
        query_size = induction_config.query_size
        batch_size, _ = get_shape_list(support_text_ids)
        # concat the support and query text and use albert to encode text
        support_text_ids = tf.reshape(support_text_ids, [batch_size * c * k, seq_len])
        support_text_mask = tf.reshape(support_text_mask, [batch_size * c * k, seq_len])
        query_text_ids = tf.reshape(query_text_ids, [batch_size * query_size, seq_len])
        query_text_mask = tf.reshape(query_text_mask, [batch_size * query_size, seq_len])
        encode_input = tf.concat([support_text_ids, query_text_ids],
                                 0)  # [batch_size * (c * k + query_size), seq_len]
        encode_mask = tf.concat([support_text_mask, query_text_mask], 0)
        self.bert_model = modeling.BertModel(config=bert_config, is_training=is_training, input_ids=encode_input,
                                             input_mask=encode_mask, scope="bert")
        # use first token embeddings as text embedding
        encode_output = tf.squeeze(self.bert_model.sequence_output[:, 0:1, :],
                                   axis=1)  # [batch_size * (c * k + query_size), hidden_size]
        support_encode = tf.reshape(encode_output[:batch_size * c * k],
                                    [batch_size, c, k,
                                     bert_config.hidden_size])  # [batch_size, c, k, hidden_size]
        query_encode = tf.reshape(encode_output[batch_size * c * k:],
                                  [batch_size, query_size,
                                   bert_config.hidden_size])  # [batch_size, query_size, hidden_size]
        self.query_encode = query_encode
        self.support_encode = support_encode
        with tf.variable_scope(scope, default_name="induction"):
            with tf.variable_scope("routing"):
                self.class_vector = induction_with_DR(support_encode)  # [batch_size, c, hidden_size]

        with tf.variable_scope(scope, default_name="relation"):
            self.relation_score = relation_model(class_vector=self.class_vector, h=induction_config.h,
                                                 query_input=query_encode)  # [batch_size, query_size, c]


def create_model(induction_config,
                 is_training,
                 support_text_ids,
                 support_text_mask,
                 query_text_ids,
                 query_text_mask,
                 seq_len,
                 bert_config,
                 query_label):
    """
    创建模型，model_fn_builder中调用
    Args:
        induction_config(ModelConfig):
        seq_len:
        is_training:
        support_text_ids: int [batch_size, c, k, seq_len]
        support_text_mask: int [batch_size, c, k, seq_len]
        query_text_ids: int [batch_size, query_size, seq_len]
        query_text_mask: int [batch_size, query_size, seq_len]
        query_label: [batch_size, query_size * c]
        bert_config:

    Returns:
         loss, query_encode, class_vector, support_encode, relation_score
         query_encode, float tensor [query_size, hidden_size]
         class_vector, float tensor [c, hidden_size]
         support_encode, float tensor [c, k, hidden_size]
         relation_score, float tensor [query_size, c]
    """

    model = InductionModel(induction_config=induction_config, is_training=is_training,
                           support_text_ids=support_text_ids,
                           support_text_mask=support_text_mask, seq_len=seq_len, query_text_ids=query_text_ids,
                           query_text_mask=query_text_mask, query_label=query_label,
                           bert_config=bert_config)
    with tf.variable_scope("loss"):
        batch_size, _ = get_shape_list(support_text_ids)
        query_label = tf.reshape(query_label, [batch_size, induction_config.query_size, induction_config.c])
        loss = tf.losses.mean_squared_error(query_label, model.relation_score)
    return loss, model.query_encode, model.class_vector, model.support_encode, model.relation_score


def model_fn_builder(induction_config,
                     bert_config,
                     init_checkpoint,
                     seq_len,
                     learning_rate=5e-5,
                     num_train_steps=100,
                     num_warmup_steps=10):
    """

    Args:
        induction_config:
        bert_config:
        init_checkpoint:
        seq_len:
        learning_rate:
        num_train_steps:
        num_warmup_steps:

    Returns:
        model_fn
    """

    def model_fn(features, labels, mode, params):
        support_text_ids = features["support_text_ids"]
        support_text_mask = features["support_text_mask"]
        query_text_ids = features["query_text_ids"]
        query_text_mask = features["query_text_mask"]
        query_label = features["query_label"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss, query_encode, class_vector, support_encode, relation_score = create_model(
            seq_len=seq_len,
            induction_config=induction_config,
            bert_config=bert_config,
            support_text_ids=support_text_ids,
            support_text_mask=support_text_mask,
            query_text_ids=query_text_ids,
            query_text_mask=query_text_mask,
            is_training=is_training,
            query_label=query_label
        )

        # init_checkpoint
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        # logging checkpoint
        # log_variables(initialized_variable_names, tvars)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

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

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics
            )

        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "query_embedding": query_encode,
                    "class_vector": class_vector,
                    "support_embedding": support_encode,
                    "relation_score": relation_score},
            )
        return output_spec

    return model_fn

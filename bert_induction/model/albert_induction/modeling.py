# encoding=utf-8
import collections
import json
import os

import numpy as np
import tensorflow as tf

from model.bert import tokenization
from model.common.base_func import get_assignment_map_from_checkpoint, log_variables, get_shape_list
from model.common.layers import self_attention_bi_lstm
from model.common.optimization import create_optimizer
from model.induction.config import ModelConfig
from model.induction.layers import induction_with_DR, relation_model


class InductionModel:
    def __init__(self, config, is_training,
                 support_ids, seq_len,
                 query_ids, query_label, scope=None):
        """

        Args:
            config (ModelConfig):
            support_ids: int64 Tensor [batch_size, c * k * seq_len * embedding_size]
            support_mask: int64 
            query_ids: int64 Tensor [batch_size, query_size * seq_len * embedding_size]
            query_label: int64 Tensor [batch_size, query_size]
            is_training:
            scope:
        """
        if not is_training:
            dropout_prob = 0.0
        else:
            dropout_prob = config.dropout_prob
        embedding_size = config.embedding_size
        c = config.c
        k = config.k
        query_size = config.query_size
        batch_size, _ = get_shape_list(support_ids)
        support_ids = tf.reshape(support_ids, [batch_size * c * k, seq_len, embedding_size])
        query_ids = tf.reshape(query_ids, [batch_size * query_size, seq_len, embedding_size])
        encode_input = tf.concat([support_ids, query_ids],
                                 0)  # [batch_size * (c * k + query_size), seq_len, embedding_size]
        encode_output = self_attention_bi_lstm(encode_input, config.hidden_size, config.attention_size, dropout_prob)
        support_encode = tf.reshape(encode_output[:batch_size * c * k],
                                    [batch_size, c, k, config.attention_size])  # [batch_size, c, k, attention_size]
        query_encode = tf.reshape(encode_output[batch_size * c * k:],
                                  [batch_size, query_size,
                                   config.attention_size])  # [batch_size, query_size, attention_size]
        self.query_encode = query_encode
        self.support_encode = support_encode
        with tf.variable_scope(scope, default_name="induction"):
            with tf.variable_scope("routing"):
                self.class_vector = induction_with_DR(support_encode)  # [batch_size, c, attention_size]

        with tf.variable_scope(scope, default_name="relation"):
            self.relation_score = relation_model(class_vector=self.class_vector, h=config.h,
                                                 query_input=query_encode)  # [batch_size, query_size, c]


def create_model(config, is_training, support_embeddings, query_embeddings, seq_len,
                 query_label):
    """
    创建模型，model_fn_builder中调用
    Args:
        seq_len:
        config (ModelConfig): induction配置
        is_training:
        support_embeddings: [batch_size, c, k, attention_size]
        query_embeddings: [batch_size, query_size, attention_size]
        query_label: [batch_size, query_size]

    Returns:
         loss, query_encode, class_vector, support_encode, relation_score
         query_encode, float tensor [query_size, attention_size]
         class_vector, float tensor [c, attention_size]
         support_encode, float tensor [c, k, attention_size]
         relation_score, float tensor [query_size, c]
    """
    model = InductionModel(config=config, seq_len=seq_len,
                           is_training=is_training,
                           support_ids=support_embeddings,
                           query_ids=query_embeddings,
                           query_label=query_label)
    with tf.variable_scope("loss"):
        query_label = tf.one_hot(query_label, depth=config.c)  # [batch_size, query_size, c]
        loss = tf.losses.mean_squared_error(query_label, model.relation_score)
    return loss, model.query_encode, model.class_vector, model.support_encode, model.relation_score


def model_fn_builder(config,
                     init_checkpoint,
                     max_seq_length,
                     learning_rate=5e-5,
                     num_train_steps=100,
                     num_warmup_steps=10):
    """

    Args:
        max_seq_length:
        config:
        init_checkpoint:
        learning_rate:
        num_train_steps:
        num_warmup_steps:

    Returns:

    """

    def model_fn(features, labels, mode, params):
        support_embedding = features["support_embedding"]
        query_embedding = features["query_embedding"]
        query_label = features["query_label"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        loss, query_encode, class_vector, support_encode, relation_score = create_model(
            seq_len=max_seq_length,
            config=config,
            is_training=is_training,
            support_embeddings=support_embedding,
            query_embeddings=query_embedding,
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
        log_variables(initialized_variable_names, tvars)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps)
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


# def fine_tune():
#     project_path = "/home/bert_few_shot"
#     model_path = os.path.join(project_path, "models")
#     model_config_fp = os.path.join(model_path, "test", "model_config.json")
#     with open(model_config_fp, 'w') as model_config_fd:
#         json.dump(ModelConfig(k=5,
#                               dropout_prob=0.1,
#                               hidden_size=128,
#                               attention_size=64,
#                               h=100).to_dict(),
#                   model_config_fd)
#     classifier(embedding_file=os.path.join(model_path, "vector", "merge_sgns_bigram_char300.txt"),
#                dict_path=os.path.join(model_path, "vector", "user_dict.txt"),
#                data_dir=os.path.join(project_path, "data", "output"),
#                config_file=model_config_fp,
#                init_checkpoint=None,
#                max_seq_length=32,
#                train_batch_size=32,
#                eval_batch_size=1,
#                predict_batch_size=1,
#                num_train_epochs=32.0,
#                output_dir=os.path.join(model_path, "test"),
#                do_train=True,
#                iterations_per_loop=100,
#                learning_rate=5e-4,
#                save_checkpoints_steps=10000,
#                use_cpu=False
#                )
#
#
# def predict(output_dir):
#     project_path = "/home/bert_few_shot"
#     model_path = os.path.join(project_path, "models")
#     model_config_fp = os.path.join(model_path, "test", "model_config.json")
#     classifier(embedding_file=os.path.join(model_path, "vector", "merge_sgns_bigram_char300.txt"),
#                dict_path=os.path.join(model_path, "vector", "user_dict.txt"),
#                data_dir=os.path.join(project_path, "data", "output", "self_support"), predict_class_num=1,
#                config_file=model_config_fp,
#                init_checkpoint=os.path.join(model_path, "test", "model.ckpt-10000"),
#                max_seq_length=64,
#                predict_batch_size=1,
#                output_dir=output_dir,
#                do_predict=True,
#                use_cpu=False
#                )
#
#
# if __name__ == "__main__":
#     import logging
#
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # fine_tune()
#     predict("/home/bert_few_shot/data/output/self_support/result")

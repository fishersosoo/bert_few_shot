# encoding=utf-8
import collections
import json
import os

import numpy as np
import tensorflow as tf

from model.bert import tokenization
from model.common.base_func import get_assignment_map_from_checkpoint
from model.common.layers import self_attention_bi_lstm
from model.common.optimization import create_optimizer
from model.induction.classifier import classifier
from model.induction.config import ModelConfig
from model.induction.layers import induction, relation_model


class CheckEmbedding:
    def __init__(self, name):
        self.is_equal = True
        self.name = name

    def __call__(self, a, b):
        if self.is_equal:
            if not np.array_equal(a, b):
                tf.logging.warning("{name} not equal!".format(name=self.name))
                self.is_equal = False


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
        encode_output = self_attention_bi_lstm(encode_input, config.hidden_size, config.attention_size, dropout_prob)
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

# coding=utf-8
import collections

import tensorflow as tf
import numpy as np

from model.bert import tokenization


class RecordCodec(object):
    @classmethod
    def write_example(cls, *args, **kwargs):
        """ 将输入数据转化成TFRecord文件
        每一个 training 迭代写成一个example"""
        raise NotImplementedError()

    @classmethod
    def build_file_base_input_fn(cls, *args, **kwargs):
        """
        读取TF_Record文件构建输入pipeline
        """
        raise NotImplementedError()


class EmbeddingsRecordCodec(RecordCodec):
    """文本会直接转换为Embeddings输入模型"""

    @classmethod
    def write_example(cls, fp_in, fp_out, max_seq_length, tokenizer=None, do_predict=False, use_exist=False):
        with tf.python_io.TFRecordWriter(fp_out) as writer:
            examples = np.load(fp_in, allow_pickle=True)
            if use_exist:
                return len(examples)
            for example in examples:
                features = collections.OrderedDict()
                support_embeddings = []
                query_embeddings = []
                for one_text in np.array(example["support_set_text"]).reshape(-1):
                    one_text = tokenization.convert_to_unicode(one_text)
                    vector = tokenizer.convert_to_vector(one_text, max_len=max_seq_length)
                    support_embeddings.append(vector)
                for one_text in example["query_set_text"]:
                    one_text = tokenization.convert_to_unicode(str(one_text))
                    vector = tokenizer.convert_to_vector(one_text, max_len=max_seq_length)
                    query_embeddings.append(vector)
                features["support_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.array(support_embeddings).reshape(-1)))
                features["query_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.array(query_embeddings).reshape(-1)))
                if do_predict:
                    features["query_label"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=example["label"].reshape(-1)))
                else:
                    features["query_label"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[0] * len(example["query_set_text"])))
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
            return len(examples)

    @classmethod
    def build_file_base_input_fn(cls, input_file, model_config, max_seq_length, batch_size, is_training,
                                 drop_remainder=False):
        k = model_config.k
        c = model_config.c
        query_size = model_config.query_size
        embedding_size = model_config.embedding_size

        def _decode_record(record):
            name_to_features = {
                "support_embedding": tf.FixedLenFeature([c * k * max_seq_length * embedding_size], tf.float32),
                "query_embedding": tf.FixedLenFeature([query_size * max_seq_length * embedding_size], tf.float32),
                "query_label": tf.FixedLenFeature([query_size], tf.int64)}
            example = tf.parse_single_example(record, name_to_features)
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.float64:
                    t = tf.cast(t, dtype=tf.float32)
                example[name] = t
            return example

        if is_training:
            drop_remainder = True

        def input_fn(params):
            batch_size=params["batch_size"]
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


class IdsRecordCodec(RecordCodec):
    """文本只会转化为ids，嵌入在模型进行"""
    @classmethod
    def write_example(cls, fp_in, fp_out, max_seq_length, tokenizer=None, do_predict=False, use_exist=False):
        with tf.python_io.TFRecordWriter(fp_out) as writer:
            examples = np.load(fp_in, allow_pickle=True)
            if use_exist:
                return len(examples)
            for example in examples:
                features = collections.OrderedDict()
                support_embeddings = []
                query_embeddings = []
                for one_text in np.array(example["support_set_text"]).reshape(-1):
                    one_text = tokenization.convert_to_unicode(one_text)
                    vector = tokenizer.convert_to_vector(one_text, max_len=max_seq_length)
                    support_embeddings.append(vector)
                for one_text in example["query_set_text"]:
                    one_text = tokenization.convert_to_unicode(str(one_text))
                    vector = tokenizer.convert_to_vector(one_text, max_len=max_seq_length)
                    query_embeddings.append(vector)
                features["support_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.array(support_embeddings).reshape(-1)))
                features["query_embedding"] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=np.array(query_embeddings).reshape(-1)))
                if do_predict:
                    features["query_label"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=example["label"].reshape(-1)))
                else:
                    features["query_label"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[0] * len(example["query_set_text"])))
                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())
            return len(examples)
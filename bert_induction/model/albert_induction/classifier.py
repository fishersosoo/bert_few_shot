# encoding=utf-8
"""运行配置，管理输入输出"""
import os
from configparser import ConfigParser

import numpy as np
import shutil
import pandas as pd
import tensorflow as tf
import re
from data_processing import get_tokenizer
from model.albert_induction.config import ModelConfig as InductionConfig
from model.albert_zh import tokenization
from model.albert_zh.modeling import BertConfig
from model.albert_induction.modeling import model_fn_builder


def file_based_input_fn_builder(input_file, is_training, batch_size, max_len, induction_config,
                                drop_remainder=True):
    name_to_features = {
        "support_text_ids": tf.FixedLenFeature([induction_config.c * induction_config.k * max_len], tf.int64),
        "support_text_mask": tf.FixedLenFeature([induction_config.c * induction_config.k * max_len], tf.int64),
        "query_text_ids": tf.FixedLenFeature([induction_config.query_size * max_len], tf.int64),
        "query_text_mask": tf.FixedLenFeature([induction_config.query_size * max_len], tf.int64),
        "query_label": tf.FixedLenFeature([induction_config.query_size, induction_config.c], tf.int64)

    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def input_fn_builder(features, batch_size):
    """
    create input_fn for prediction
    Args:
        features:
            "support_text_ids": int array with shape [num_examples, c, k, seq_len]
            "support_text_mask": int array with shape [num_examples, c, k, seq_len]
            "query_text_ids": int array with shape [num_examples, query_size, seq_len]
            "query_text_mask": int array with shape [num_examples, query_size, seq_len]

    Returns:
        input_fn
    """

    support_text_ids = np.array(features["support_text_ids"])
    support_text_mask = np.array(features["support_text_mask"])
    query_text_ids = np.array(features["query_text_ids"])
    query_text_mask = np.array(features["query_text_mask"])
    num_examples, c, k, seq_len = support_text_ids.shape
    _, query_size, _ = query_text_ids.shape
    query_label = np.zeros([num_examples, query_size, c]).astype(int)

    def input_fn(params):
        d = tf.data.Dataset.from_tensor_slices({
            "support_text_ids": tf.constant(
                support_text_ids.reshape([num_examples, c * k * seq_len]), shape=[num_examples, c * k * seq_len],
                dtype=tf.int32
            ),
            "support_text_mask": tf.constant(
                support_text_mask.reshape([num_examples, c * k * seq_len]), shape=[num_examples, c * k * seq_len],
                dtype=tf.int32
            ),
            "query_text_ids": tf.constant(
                query_text_ids.reshape([num_examples, query_size * seq_len]),
                shape=[num_examples, query_size * seq_len],
                dtype=tf.int32
            ),
            "query_text_mask": tf.constant(
                query_text_mask.reshape([num_examples, query_size * seq_len]),
                shape=[num_examples, query_size * seq_len],
                dtype=tf.int32
            ),
            "query_label": tf.constant(
                query_label.reshape([num_examples, query_size * c]),
                shape=[num_examples, query_size * c],
                dtype=tf.int32
            )
        })
        d = d.batch(batch_size=1, drop_remainder=True)
        return d

    return input_fn


class Classifier(object):
    def __init__(self):
        self.estimator = None
        self._tokenizer = None
        self._bert_config = None
        self._induction_config = None
        self._run_config = tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    def _get_meta(self, model_dir):
        """
        读取模型目录的meta信息
        Args:
            model_dir:

        Returns:

        """
        path_re = re.compile(r".*_path")
        meta_dict = dict()
        config = ConfigParser()
        config.read(os.path.join(model_dir, "meta.ini"))
        for section in config.sections():
            meta_dict[section] = dict()
            for name, value in config.items(section):
                if re.match(path_re, name) is not None:
                    meta_dict[section][name] = os.path.join(model_dir, value)
                else:
                    meta_dict[section][name] = value
        return meta_dict

    def load(self, model_dir):
        """
        加载训练好的模型，设置estimator
        Args:
            model_dir: 模型目录

        Returns:

        """
        self._model_dir = model_dir
        self._meta = self._get_meta(model_dir=model_dir)
        self._tokenizer = get_tokenizer(self._meta)
        self._induction_config = InductionConfig.from_json_file(self._meta["induction"]["config_path"])
        self._bert_config = BertConfig.from_json_file(self._meta["bert"]["config_path"])

    def predict(self, support_text, query_text, batch_size=20):
        """

        Args:
            support_text: string array [k]
            query_text: string array

        Returns:
            ret: dict
            "score": float array same len as query_text
        """
        self._induction_config.c = 1
        self._induction_config.query_size = batch_size
        self._induction_config.k = len(support_text)
        model_fn = model_fn_builder(
            induction_config=self._induction_config,
            bert_config=self._bert_config,
            init_checkpoint=self._meta["model"]["checkpoint_path"],
            seq_len=int(self._meta["tokenizer"]["max_len"])
        )
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=self._model_dir,
            config=self._run_config)
        features = self._build_features(query_text, support_text)
        input_fn = input_fn_builder(features, 1)
        results = [one for one in self.estimator.predict(input_fn=input_fn)]
        ret = self._parse_results(results)
        ret_drop = dict()
        for k, v in ret.items():
            ret_drop[k] = v[:len(query_text)]
        return ret_drop

    def _build_features(self, query_text, support_text):
        features = dict()
        features["query_text_ids"] = []  # [sample_num ,query_size, max_len]
        features["query_text_mask"] = []
        features["query_label"] = []  # [sample_num ,query_size, c]
        padding_query_label = [[0] * self._induction_config.c] * self._induction_config.query_size
        support_ids = []
        support_mask = []
        for text in support_text:
            ids, mask = self._tokenizer.convert_to_vector(tokenization.convert_to_unicode(text),
                                                          int(self._meta["tokenizer"]["max_len"]))
            support_ids.append(ids)
            support_mask.append(mask)
        padding_ids = []
        padding_mask = []
        for i in range(self._induction_config.k):
            empty_ids, empty_mask = self._tokenizer.convert_to_vector("",
                                                                      int(self._meta["tokenizer"]["max_len"]))
            padding_ids.append(empty_ids)
            padding_mask.append(empty_mask)
        single_example_ids = [support_ids]
        single_example_mask = [support_mask]
        while len(single_example_ids) < self._induction_config.c:
            # 　padding empty class
            single_example_ids.append(padding_ids)
            single_example_mask.append(padding_mask)
        query_size = self._induction_config.query_size
        sample_num = np.ceil(len(query_text) / query_size).astype(int)
        features["support_text_ids"] = [single_example_ids] * sample_num  # [sample_num, c, k, max_len]
        features["support_text_mask"] = [single_example_mask] * sample_num
        for sample_index in range(sample_num):
            single_query_ids = []
            single_query_mask = []
            for query_index in range(query_size):
                if sample_index * query_size + query_index >= len(query_text):
                    text = ""
                else:
                    text = query_text[sample_index * query_size + query_index]
                ids, mask = self._tokenizer.convert_to_vector(
                    tokenization.convert_to_unicode(text),
                    int(self._meta["tokenizer"]["max_len"]))
                single_query_ids.append(ids)
                single_query_mask.append(mask)
            features["query_text_mask"].append(single_query_mask)
            features["query_text_ids"].append(single_query_ids)
            features["query_label"].append(padding_query_label)
        return features

    def _parse_results(self, results):

        query_size = self._induction_config.query_size
        ret = {"score": []}
        for result in results:
            for query_index in range(query_size):
                ret["score"].append(result["relation_score"][query_index][0])
        return ret

    def train(self,
              training_data_path,
              induction_config,
              bert_config,
              tokenizer_name,
              vocab_path,
              file_based_convert_examples_to_features_fn,
              use_existed=True,
              save_dir=None,
              max_len=128,
              init_checkpoint=None,
              batch_size=32,
              num_train_epochs=3.0,
              learning_rate=5e-5,
              warmup_proportion=0.1,
              ):
        """

        Args:
            training_data_path:
            induction_config:
            bert_config:
            tokenizer_name:
            vocab_path:
            file_based_convert_examples_to_features_fn:
                convert train/test data to "TFrecord" file
`               def file_based_convert_examples_to_features_fn(training_data_path, tokenizer, save_dir, max_len, induction_config):

            save_dir:
            max_len:
            init_checkpoint:
            batch_size:
            num_train_epochs:
            learning_rate:
            warmup_proportion:

        Returns:

        """
        meta_dict = {"tokenizer": {"name": tokenizer_name, "vocab_path": vocab_path, "max_len": max_len}}
        tokenizer = get_tokenizer(meta_dict)
        sample_num = file_based_convert_examples_to_features_fn(training_data_path, tokenizer, save_dir, max_len,
                                                                induction_config, use_existed=use_existed)
        num_train_steps = int(sample_num / batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

        input_fn = file_based_input_fn_builder(os.path.join(save_dir, "train.tf_record"),
                                               is_training=True,
                                               batch_size=batch_size,
                                               max_len=max_len,
                                               induction_config=induction_config)
        model_fn = model_fn_builder(induction_config=induction_config,
                                    bert_config=bert_config,
                                    init_checkpoint=init_checkpoint,
                                    seq_len=max_len,
                                    learning_rate=learning_rate,
                                    num_train_steps=num_train_steps,
                                    num_warmup_steps=num_warmup_steps
                                    )
        run_config = tf.estimator.RunConfig(
            session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=save_dir,
            config=run_config)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", sample_num)
        tf.logging.info("  Batch size = %d", batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        estimator.train(input_fn, max_steps=num_train_steps)

        meta_dict["model"] = {"checkpoint_path": os.path.split(estimator.latest_checkpoint())[1]}
        self.save(save_dir, meta_dict, induction_config, bert_config)

    def save(self, save_dir, meta_dict=None, induction_config=None, bert_config=None):
        if meta_dict is None:
            meta_dict = {}
        if induction_config is not None:
            meta_dict["induction"] = {"config_path": "induction_config.json"}
            with open(os.path.join(save_dir, "induction_config.json"), mode='w') as fd:
                fd.write(induction_config.to_json_string())
        if bert_config is not None:
            meta_dict["bert"] = {"config_path": "bert_config.json"}
            with open(os.path.join(save_dir, "bert_config.json"), mode='w') as fd:
                fd.write(bert_config.to_json_string())
        if meta_dict is not None:
            shutil.copy2(meta_dict["tokenizer"]["vocab_path"], save_dir)
            meta_dict["tokenizer"]["vocab_path"] = os.path.split(meta_dict["tokenizer"]["vocab_path"])[1]
            config_parser = ConfigParser()
            for section, v in meta_dict.items():
                config_parser.add_section(str(section))
                for option, value in v.items():
                    config_parser.set(section, str(option), str(value))
            with open(os.path.join(save_dir, "meta.ini"), encoding='utf-8', mode='w') as meta_fd:
                config_parser.write(meta_fd)

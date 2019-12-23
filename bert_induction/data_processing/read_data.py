import collections
import json
import logging
import os

import tensorflow as tf
import numpy as np
import pandas as pd
from abc import abstractmethod

from bert import tokenization

log = logging.getLogger("data_process")
# sh = logging.StreamHandler()
# sh.setFormatter(format_str)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


class Dataset(object):
    @abstractmethod
    def read_raw_data_file(self, raw_fp, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def train_test_split(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_training_examples(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_test_examples(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def write_example(self, *args, **kwargs):
        """
        将输入数据转化成TFRecord文件
        每一个 training 迭代写成一个example
        Args:
            fp_in:
            fp_out:
            max_seq_length:
            tokenizer:
            do_predict:

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def build_file_base_input_fn(self, *args, **kwargs):
        """
        读取TF_Record构建输入pipeline
        Args:
            input_file:
            params:
            is_training:
            drop_remainder:

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def convert_examples_to_features(self, examples, label_list, params,
                                     tokenizer):
        """
        构建成features，配合build_input_fn使用
        Args:
            examples:
            label_list:
            params:
            tokenizer:

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def build_input_fn(self, *args, **kwargs):
        raise NotImplementedError()


class OnlineShoppingData(Dataset):
    def convert_examples_to_features(self, examples, label_list, params, tokenizer):

        features = []
        pass

    def build_input_fn(self, features, config, max_seq_length, is_training, drop_remainder):
        # support_embedding=[]
        # query_embedding=[]
        # query_label=[]
        # for feature in features:
        #     support_embedding.append(feature["support_embedding"])
        #     query_embedding.append(feature["query_embedding"])
        #     query_label.append(feature["query_label"])
        #
        # def input_fn(params):
        #     batch_size = params["batch_size"]
        #     num_examples = len(features)
        #     d
        pass

    def __init__(self):
        pass

    def read_raw_data_file(self, raw_fp, *args, **kwargs):
        """
        处理源数据集
        Args:
            path: 数据集路径

        Returns:
            dataset. DataFrame with columns ["cat", "label", "review", "class"]
                cat, 中文类别名称
                label, 评价正负面
                review, 评论文本
                class, 类别id
            class_info. DataFrame with columns ["cat", "label", "class"]
                字段意义同上，用于记录"cat", "label" 和 "class" 的对应关系
        """
        dataset = pd.read_csv(raw_fp, encoding="utf-8")
        cats = dataset["cat"].drop_duplicates()
        labels = dataset["label"].drop_duplicates()
        info = {"class": [], "cat": [], "label": []}
        index = 0
        for cat in cats:
            for label in labels:
                info["class"].append(index)
                info["cat"].append(cat)
                info["label"].append(label)
                index += 1
        class_info = pd.DataFrame(data=info)
        dataset = pd.merge(dataset, class_info, on=["label", "cat"])
        return dataset, class_info

    def train_test_split(self, dataset, class_info, training_set_info=None, training_set_class_num=None, *args,
                         **kwargs):
        if training_set_info is None:
            log.info(
                "No saved train test split info. select {training_set_class_num} training classes randomly.".format(
                    training_set_class_num=training_set_class_num))
            training_set_info = class_info.sample(training_set_class_num)
        else:
            log.info(
                "Last train test split result is used.".format(
                    training_set_class_num=training_set_class_num))
        training_set = dataset[dataset["class"].isin(training_set_info["class"])]
        test_set_info = class_info[
            np.logical_not(class_info["class"].isin(training_set_info["class"]))]
        test_set = dataset[np.logical_not(dataset["class"].isin(training_set_info["class"]))]
        return training_set, training_set_info, test_set, test_set_info

    def get_training_examples(self, training_set, c, k, query_per_class, training_iter_num):
        """
        构建元训练的训练集（支撑集和预测集）
        Args:
            training_iter_num:
            query_per_class: 每次运行每个类选取多少个query
            training_set:
            c:
            k:

        Returns:
            examples. numpy array, with shape [training_iter_num]
                each example {
                support_set_text. string array with shape [c, k]
                query_set_text. string array with shape [c * query_per_class]
                label: int array with shape [c * query_per_class]
                }
        """
        examples = []
        for training_iter in range(training_iter_num):
            support_set_text = []
            query_set = pd.DataFrame()
            # pick c class
            for class_id, one_class in enumerate(training_set["class"].unique().sample(c)):
                class_samples = training_set[training_set["class"] == one_class]
                # select k support samples
                support_text = class_samples.sample(k)["review"].to_list()
                support_set_text.append(support_text.to_list())
                # select query samples
                query = class_samples.sample(query_per_class)
                query["class_id"] = class_id
                query_set = query_set.append(query, ignore_index=True)
            query_set = query_set.sample(frac=1.)
            example = {
                "support_set_text": support_set_text,
                "query_set_text": query_set["review"].to_list(),
                "label": query_set["label"].to_list()
            }
            examples.append(example)
        return examples

    def get_test_examples(self, test_set, query_size, sample_per_class, c, k):
        """
        生成测试数据
        先从测试集中每类选取sample_per_class个样本
        计算所需运行的次数
        每次运行选择c * k个样本
        对于每个query，随机从样本中选取k个作为support set
        Args:
            query_size:
            test_set:
            sample_per_class:
            c:
            k:

        Returns:
            query_set_df. DataFrame with column ["cat", "label", "review", "class","class_id"]
                class_id, range from 0 to class_num.
            examples. numpy array, with shape [training_iter_num]
                each example {
                support_set_text. string array with shape [c, k]
                query_set_text. string array with shape [query_size]
                }
        """
        class_labels = test_set["class"].unique()
        class_num = len(class_labels)
        query_set_df = pd.DataFrame()
        for class_id, class_label in enumerate(class_labels):
            if len(test_set[test_set["class"] == class_label]) <= sample_per_class:
                selected_examples = test_set[test_set["class"] == class_label].sample(sample_per_class)
            else:
                selected_examples = test_set[test_set["class"] == class_label]
            selected_examples["class_id"] = class_id
            query_set_df = query_set_df.append(selected_examples, ignore_index=True)
        iter_per_run = int(np.ceil(class_num / c))
        examples = []
        for run_id in range(int(np.ceil(len(query_set_df) / query_size))):
            # get query text
            run_id_start = run_id * query_size
            run_id_end = run_id_start + query_size
            if run_id_end <= len(query_set_df):
                query_text = query_set_df[run_id_start:run_id_end]["review"].to_list()
            else:
                # padding
                query_text = query_set_df[run_id_start:]["review"].to_list()
                query_text.extend([""] * (query_size - len(query_text)))
                for iter_index in range(iter_per_run):
                    # build example
                    example = {"query_set_text": query_text, "support_set_text": []}
                    for class_index in range(c):
                        class_id = iter_index * c + class_index
                        if class_id >= class_num:
                            # padding
                            example["support_set_text"].append([""] * k)
                        else:
                            example["support_set_text"].append(
                                test_set[test_set["class"] == class_labels[class_id]].sample(k)["review"].to_list())
                    examples.append(example)
        return query_set_df, examples

    def write_example(self, fp_in, fp_out, max_seq_length, tokenizer=None, do_predict=False):
        with tf.python_io.TFRecordWriter(fp_out) as writer:
            examples = np.load(fp_in, allow_pickle=True)
            for example in examples:
                features = collections.OrderedDict()
                support_embeddings = []
                query_embeddings = []
                for one_text in example["support_set_text"].reshape(-1):
                    one_text = tokenization.convert_to_unicode(one_text)
                    vector = tokenizer.convert_to_vector(one_text, max_len=max_seq_length)
                    support_embeddings.append(vector)
                for one_text in example["query_set_text"]:
                    one_text = tokenization.convert_to_unicode(one_text)
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

    def build_file_base_input_fn(self, input_file, config, max_seq_length, is_training, drop_remainder):
        k = config.k
        c = config.c
        query_size = config.query_size
        embedding_size = config.embedding_size

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

# class TNEWSData():
#     @classmethod
#     def _process_dataset(cls, path):
#         with open(path, encoding="UTF-8") as source_file:
#             dataset = {"sample_id": [], "class_id": [], "class_name": [], "text": []}
#             for line in source_file:
#                 sample_id, class_id, class_name, text, tag = line.split("_!_")
#                 dataset["sample_id"].append(int(sample_id))
#                 dataset["class_id"].append(int(class_id))
#                 dataset["class_name"].append(class_name)
#                 dataset["text"].append(text)
#
#
# class NSData():
#     @classmethod
#     def _process_dataset(cls, path):
#
#         with open(path, encoding="utf-8") as f:
#             guide_df = {"guide_id": [], "label": [], "title": []}
#             guides = json.load(f)
#             for guide in guides:
#                 guide_df["guide_id"].append(guide["guide_id"])
#                 guide_df["title"].append(guide["title"])
#                 guide_df["label"].append(guide["label"])
#             return pd.DataFrame(guide_df)
#
#     def __init__(self, path, k=5):
#         self.guides = self._process_dataset(path)
#         self.labels = []
#         for i, guide in self.guides.iterrows():
#             self.labels.extend(guide["label"])
#         self.labels = set(self.labels)
#         self.k = k
#
#     def generate_test_data(self, output_dir):
#         """
#
#         Returns:
#             query_input, string array with shape [run_size]
#             support_input. string array with shape [run_size, k]
#             query_set_df. DataFrame with column ["title", 0, 1, 2, ... , class_num]
#             class_index, class_id to class_name
#             support_set_df. DataFrame with column ["title", "labels"]
#         """
#         query_input = []
#         support_input = []
#         predict_labels = []
#         support_set_texts = []  # [class_num, k]
#         df = self.guides[["title"]].copy()
#         for label in self.labels:
#             has_label = self.guides["label"].apply(lambda labels: label in labels)
#             count = np.sum(has_label)
#             log.info("{label}:{count}".format(label=label, count=count))
#             if count > 25:
#                 predict_labels.append(label)
#         log.info("label to predict: {label}".format(label=predict_labels))
#         class_num = len(predict_labels)
#         for label_id, label in enumerate(predict_labels):
#             has_label = self.guides["label"].apply(lambda labels: label in labels)
#             df[label_id] = has_label.astype(int)
#             support_set_text = self.guides[has_label].sample(self.k)["title"].to_list()
#             support_set_texts.append(support_set_text)
#         support_set_texts_flattent = np.array(support_set_texts).reshape(-1)
#         query_set_df = df[np.logical_not(df["title"].isin(support_set_texts_flattent))]
#         for query in query_set_df["title"]:
#             for class_id in range(class_num):
#                 query_input.append(query)
#                 support_input.append(support_set_texts[class_id])
#         info_dict = {"predict_labels": predict_labels, "query_set_df": query_set_df.to_dict()}
#         with open(os.path.join(output_dir, "info.json"), 'w', encoding='utf-8') as f:
#             json.dump(info_dict, f)
#         np.save(os.path.join(output_dir, "support_text"), np.array(support_input))
#         np.save(os.path.join(output_dir, "query_text"), np.array(query_input))


# if __name__ == '__main__':
#     # d = NSData("/home/bert_few_shot/data/source/NS/source.json", 5)
#     # d.generate_test_data("/home/bert_few_shot/data/output/NS/predict")

import collections
import os
from collections import __init__

import numpy as np
import pandas as pd
import tensorflow as tf

from data_processing.data_set import Dataset, log
from model.bert import tokenization


class OnlineShoppingData(Dataset):
    def convert_examples_to_features(self, examples, label_list, params, tokenizer):
        raise NotImplementedError()

    def build_input_fn(self, features, config, max_seq_length, is_training, drop_remainder):
        raise NotImplementedError()

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

    def train_test_split(self, dataset, class_info, training_set_info=None, training_set_cat_num=7, *args,
                         **kwargs):
        if training_set_info is None:
            log.info(
                "No saved train test split info. select {training_set_class_num} training classes randomly.".format(
                    training_set_class_num=training_set_cat_num))
            training_cats = class_info["cat"].sample(training_set_cat_num)
            training_set_info = class_info[class_info["cat"].isin(training_cats)]
        else:
            log.info(
                "Last train test split result is used.".format(
                    training_set_class_num=training_set_cat_num))
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
            picked_classes = np.random.choice(training_set["class"].unique(), c, False)
            for class_id, one_class in enumerate(picked_classes):
                class_samples = training_set[training_set["class"] == one_class]
                # select k support samples
                support_text = class_samples.sample(k)["review"].to_list()
                support_set_text.append(support_text)
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
            if len(test_set[test_set["class"] == class_label]) > sample_per_class:
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

    def build_file_base_input_fn(self, input_file, config, max_seq_length, batch_size, is_training,
                                 drop_remainder=False):
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


def build_c_way_k_shot(raw_data_fp, c, k, query_per_class, training_iter_num, training_set_cat_num, output_dir):
    data_name = "online_shopping_10_cats"
    data_dir = os.path.join(output_dir, data_name, "{c}-way {k}-shot".format(c=c, k=k))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    training_info_fp = os.path.join(data_dir, "train_info.csv")
    training_info = None
    if training_info_fp is not None and os.path.exists(training_info_fp):
        training_info = pd.read_csv(training_info_fp, encoding="utf-8", index_col=None)
    data_set = OnlineShoppingData()
    data_df, class_info = data_set.read_raw_data_file(raw_data_fp)
    training_df, training_info, test_df, test_info = data_set.train_test_split(data_df,
                                                                               class_info,
                                                                               training_info,
                                                                               training_set_cat_num=training_set_cat_num)
    if training_info_fp is not None:
        training_info.to_csv(training_info_fp, encoding="utf-8", index=False)
    training_examples = data_set.get_training_examples(training_df, c, k, query_per_class=query_per_class,
                                                       training_iter_num=training_iter_num)
    np.save(os.path.join(data_dir, "training_examples.npy"), training_examples, allow_pickle=True)
    query_set_df, test_examples = data_set.get_test_examples(test_df, c * query_per_class, 1000, c, k)
    np.save(os.path.join(data_dir, "test_examples.npy"), test_examples, allow_pickle=True)
    query_set_df.to_csv(os.path.join(data_dir, "test.csv"), encoding="utf-8", index=False)


def main():
    build_c_way_k_shot(
        raw_data_fp="/home/bert_few_shot/data/source/online_shopping_10_cats/online_shopping_10_cats.csv",
        output_dir="/home/bert_few_shot/data",
        c=2,
        k=5,
        query_per_class=10,
        training_iter_num=3000,
        training_set_cat_num=7)


if __name__ == '__main__':
    main()

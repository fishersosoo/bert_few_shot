import collections
import os
from collections import __init__

import numpy as np
import pandas as pd
import tensorflow as tf

from data_processing.data_set import Dataset, log
from model.albert_zh import tokenization


def file_based_convert_examples_to_features_fn(training_data_path,
                                               tokenizer,
                                               save_dir,
                                               max_len,
                                               induction_config, use_existed=False):
    """
    构建策略，总样本条数=记录数/query_size * 2，这样大多数的记录都会出现在query里面
    每条样本选取c个类别，按照 (c-1):1 的比例分别从c个类别中和c个类别外选取query
    query_label 为[query_size, c]的数组，query_label[i,j]=1,if i-th record belongs to j-th class
    Args:
        use_existed:
        training_data_path:
        tokenizer:
        save_dir:
        max_len:
        induction_config:

    Returns:

    """
    log.info("building tf record...")
    df = pd.read_csv(training_data_path, encoding="utf-8", index_col=None)
    c, k, query_size = induction_config.c, induction_config.k, induction_config.query_size
    example_num = int(len(df) / query_size * 2)
    if os.path.exists(os.path.join(save_dir, "train.tf_record")) and use_existed:
        log.info(str(os.path.join(save_dir, "train.tf_record")) + " existed")
        log.info("use_existed: True")
        return example_num
    log.info("************")
    log.info(str(example_num) + " examples")
    in_class_query_num = np.ceil(query_size * (c - 1) / c).astype(int)
    out_class_query_num = query_size - in_class_query_num
    all_class_id = df["class"].unique()
    with tf.python_io.TFRecordWriter(os.path.join(save_dir, "train.tf_record")) as writer:
        for example_id in range(example_num):
            if example_id % int(example_num / 10) == 0:
                log.info("{percent:.2%}".format(percent=example_id / example_num))
            features = collections.OrderedDict()
            all_query_label = []
            all_query_text_ids = []
            all_query_text_mask = []
            all_support_text_ids = []
            all_suppoert_text_mask = []
            # select C classes
            selected_classes = np.random.choice(all_class_id, c, False)
            # select query_size instances for query
            query_instances = df[df["class"].isin(selected_classes)].sample(in_class_query_num)
            query_instances = query_instances.append(df.sample(out_class_query_num), ignore_index=True)
            query_instances = query_instances.sample(frac=1.0)
            for instance_id, row in query_instances.iterrows():
                text = row["review"]
                text = tokenization.convert_to_unicode(text)
                ids, mask = tokenizer.convert_to_vector(text, max_len)
                all_query_text_ids.append(ids)
                all_query_text_mask.append(mask)
                all_query_label.append(np.equal(selected_classes, row["class"]).astype(int))
            # select k instances for each class
            for selected_class in selected_classes:
                support_ids, support_masks = [], []
                support_reviews = df[df["class"] == selected_class]["review"].sample(k)
                for review in support_reviews:
                    review = tokenization.convert_to_unicode(review)
                    ids, mask = tokenizer.convert_to_vector(review, max_len)
                    support_ids.append(ids)
                    support_masks.append(mask)
                all_support_text_ids.append(support_ids)
                all_suppoert_text_mask.append(support_masks)
            features["support_text_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=np.array(all_support_text_ids).reshape(-1)
                )
            )
            features["support_text_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=np.array(all_suppoert_text_mask).reshape(-1)
                )
            )
            features["query_text_ids"] = tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=np.array(all_query_text_ids).reshape(-1)
                )
            )
            features["query_text_mask"] = tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=np.array(all_query_text_mask).reshape(-1)
                )
            )
            features["query_label"] = tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=np.array(all_query_label).reshape(-1)
                )
            )
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    log.info("************")
    return example_num


class OnlineShoppingData(Dataset):
    """
    数据集一共包含10种商品，选取6种商品用于训练，剩下4种商品用于测试
    """

    def build_train_test(self, source_path, output_data_dir, **kwargs):
        log.info("building train test data...")

        training_fp = os.path.join(output_data_dir, "train.csv")
        test_fp = os.path.join(output_data_dir, "test.csv")
        if kwargs.get("use_existed", False) and \
                os.path.exists(training_fp) and \
                os.path.exists(test_fp):
            return training_fp, test_fp

        class_info = pd.read_csv(os.path.join(source_path, "class_info.csv"), encoding="utf-8", index_col=None)
        source_data = pd.read_csv(os.path.join(source_path, "online_shopping_10_cats_id.csv"), encoding="utf-8",
                                  index_col=None)
        training_set_cat_num = kwargs.get("training_cat_num", 6)
        training_cats = np.random.choice(class_info["cat"].unique(), training_set_cat_num, False)

        def map_func(cat):
            if cat in training_cats:
                return 1
            else:
                return 0

        class_info["for_training"] = class_info["cat"].map(map_func)
        training_set = source_data[source_data["cat"].isin(training_cats)]
        test_set = source_data[np.logical_not(source_data["cat"].isin(training_cats))]
        training_set.to_csv(training_fp, index=False, encoding="utf-8")
        test_set.to_csv(test_fp, index=False, encoding="utf-8")
        class_info.to_csv(os.path.join(output_data_dir, "class_info.csv"),
                          index=False, encoding="utf-8")
        return training_fp, test_fp

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
        dataset = pd.read_csv(raw_fp, encoding="utf-8", index_col=None)
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

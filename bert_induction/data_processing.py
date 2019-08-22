# coding=utf-8
import pandas as pd
import tensorflow as tf
import numpy as np


def process_dataset(path):
    """
    处理源数据集
    Args:
        path: 数据集路径

    Returns:

    """
    dataset = pd.read_csv(path)
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


def split_dataset(dataset, frac):
    """
    将数据集划分未训练集和验证集

    Args:
        dataset: 数据集
        frac: 用于训练集比例

    Returns:
        train: 训练集dataframe
        validation: 验证集dataframe
    """
    msk = np.random.rand(len(dataset)) < frac
    train = dataset[msk]
    validation = dataset[~msk]
    return train, validation

# def build_training_tfrecord(df, save_path, c, k, query_size, size):
#     """
#     构建训练的tfrecord文件
#
#     Args:
#         size: 文件包含多少条数据
#         save_path:保存的路径
#         df:数据集dataframe，包含class列和text列
#         c:每轮有多少个类
#         k:每个类有多少个样本
#         query_size:每轮训练的每个类预测多少个样本
#
#     Notes:
#         一轮训练需要的样本数为c*k+c*query_size
#     Returns:
#
#     """
#     classes = df["class"].drop_duplicates()
#     with tf.python_io.TFRecordWriter(save_path) as tfrecord_wrt:
#         for i in range(size):
#             selected_classes = classes.sample(c).to_list()
#             for selected_class in selected_classes:
#                 selected_support_samples = df[df["class"] == selected_classes].sample(k)
#                 selected_query_sample = df[df["class"] == selected_classes].sample(query_size)
#
#             df[df['class'].isin(classes.sample(2))].sample(5)
#             ids = tf.train.FeatureList(feature=
#                                        [tf.train.Feature(int64_list=tf.train.Int64List(value=[]),])
#             tf.train.Example()
#             tf.train.FeatureLists
#             pass

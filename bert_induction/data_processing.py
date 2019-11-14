# coding=utf-8
import os
import logging
import numpy as np
import pandas as pd
import json

log = logging.getLogger("data_process")
# sh = logging.StreamHandler()
# sh.setFormatter(format_str)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


def conver_to_onehot(y, class_num):
    return np.eye(class_num)[y.reshape(-1)].astype(int)


def process_dataset(path):
    """
    处理源数据集
    Args:
        path: 数据集路径

    Returns:
        dataset: review, label, cat ,class.
         class_info：label, cat ,class. 表示每个class对应的label和cat
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


def log_array(array, name):
    log.info("{name}\nshape: {shape}\ndtype: {dtype}\n\n".format(
        name=name,
        shape=np.array(array).shape,
        dtype=np.array(array).dtype
    ))


class OnlineShoppingData():

    def params(self):
        return {"c": self.c, "k": self.k, "query_size_per_class": self.query_size_per_class}

    @classmethod
    def _process_dataset(cls, path):
        """
        处理源数据集
        Args:
            path: 数据集路径

        Returns:
            path: 数据文件路径
        """
        dataset = pd.read_csv(path, encoding="utf-8")
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

    def __init__(self, input_csv, c=2, k=5, query_size_per_class=5):
        log.info("{num} examples per training iter".format(num=c * (k + query_size_per_class)))
        self.input_csv = input_csv
        self.dataset, self.class_info = self._process_dataset(input_csv)
        self.c = c
        self.k = k
        self.query_size_per_class = query_size_per_class

    def choose_train_test_class(self, training_set_class_num):
        self.training_set_class = self.class_info.sample(training_set_class_num)
        self.training_set = self.dataset[self.dataset["class"].isin(self.training_set_class["class"])]
        self.test_set_class = self.class_info[
            np.logical_not(self.class_info["class"].isin(self.training_set_class["class"]))]
        self.test_set = self.dataset[np.logical_not(self.dataset["class"].isin(self.training_set_class["class"]))]

    def generate_training_data(self, training_iter_num):
        """

        Args:
            training_iter_num:

        Returns:
            support_set_text = []
            query_set_text = []
            query_set_label = []
        """
        support_set_text = []
        query_set_text = []
        query_set_label = []
        for training_iter in range(training_iter_num):
            selected_class = self.training_set_class.sample(self.c)
            # 随机选c个类，每个类随机选k个样本和query_size_per_class个样本
            iter_support_set_text = []
            one_iter_query_set_df = pd.DataFrame(columns=["review", "label"])
            for index, class_label in enumerate(selected_class["class"]):
                iter_support_set_text.append(
                    self.training_set[self.training_set["class"] == class_label].sample(self.k)[
                        "review"].to_list())

                query_sample = pd.DataFrame(columns=["review", "label"])
                query_sample["review"] = \
                    self.training_set[self.training_set["class"] == class_label].sample(self.query_size_per_class)[
                        'review']
                query_sample["label"] = index
                one_iter_query_set_df = one_iter_query_set_df.append(query_sample)
            one_iter_query_set_df = one_iter_query_set_df.sample(frac=1.0)
            query_set_text.append(one_iter_query_set_df["review"].to_list())
            query_set_label.append(conver_to_onehot(np.array(one_iter_query_set_df["label"].to_list()), self.c))
            support_set_text.append(iter_support_set_text)
        return support_set_text, query_set_text, query_set_label

    def generate_test_data(self, sample_per_class):
        """

        Args:
            sample_per_class:

        Returns:
            support_text, string array with shape [test_class_num, self.k]
            text, string array with shape [sample_per_class * test_class_num]
            label, int array with shape [sample_per_class * test_class_num]
        """
        query_set_df = None
        support_set_df = pd.DataFrame()
        support_text = []
        for class_index, one_class in enumerate(self.test_set_class["class"]):
            class_sample = self.test_set[self.test_set["class"] == one_class]
            class_support_sample = class_sample.sample(self.k)
            # 选择支撑集
            class_support_sample["class_index"] = class_index
            support_set_df = support_set_df.append(class_support_sample, ignore_index=True)
            support_text.append(class_support_sample["review"].to_list())
            # 选择验证集
            if len(class_sample) < sample_per_class:
                selected_sample = class_sample
            else:
                selected_sample = class_sample.sample(sample_per_class)
            selected_sample["class_index"] = class_index
            if query_set_df is None:
                query_set_df = selected_sample
            else:
                query_set_df = query_set_df.append(selected_sample, ignore_index=True)
        # print(query_set_df)
        query_set_df = query_set_df.sample(frac=1.0)
        query_set_text = query_set_df["review"].to_list()

        # 计算迭代次数
        class_size = len(self.test_set_class["class"])
        sample_size = sample_per_class * class_size
        query_size = self.query_size_per_class * self.c
        run_size = np.ceil(sample_size / query_size).astype(int)
        iter_num_per_run = np.ceil(class_size / self.c).astype(int)  # 一次查询需要多少个iter
        iter_size = run_size * iter_num_per_run

        # build padding for query set
        padding_size = run_size * query_size - sample_size
        query_set_text.extend([""] * padding_size)
        # build padding for support set
        padding_size = iter_num_per_run * self.c - class_size  # 需要padding的类数量
        for i in range(padding_size):
            support_text.append([""] * self.k)

        all_support_text = []
        all_query_text = []
        for run_id in range(run_size):
            for iter_index in range(iter_num_per_run):
                iter_support = []
                for c_index in range(self.c):
                    iter_support.append(support_text[iter_index * self.c + c_index])
                all_support_text.append(iter_support)
                iter_query = query_set_text[run_id:run_id + query_size]
                all_query_text.append(iter_query)
        return all_support_text, all_query_text, query_set_df["class"].to_list(), query_set_df, support_set_df


def save_list(data, fp):
    array = np.array(data)
    log.info("saving {dtype} array with shape {shape} to {fp}".format(dtype=array.dtype, shape=array.shape, fp=fp))
    np.save(fp, array)


def check_dir(fp):
    if not os.path.exists(fp):
        os.mkdir(fp)


def generate_data(input_csv, output_dir):
    check_dir(output_dir)
    training_dir = os.path.join(output_dir, "train")
    data = OnlineShoppingData(input_csv, c=2, k=5, query_size_per_class=5)
    with open(os.path.join(output_dir, "data_param.json"), "w") as fp:
        json.dump(data.params(), fp)
    data.choose_train_test_class(15)
    support_set_text, query_set_text, query_set_label = data.generate_training_data(training_iter_num=100)
    check_dir(training_dir)
    save_list(support_set_text, os.path.join(training_dir, "support_text"))
    save_list(query_set_text, os.path.join(training_dir, "query_text"))
    save_list(query_set_label, os.path.join(training_dir, "query_label"))

    test_dir = os.path.join(output_dir, "predict")
    check_dir(test_dir)
    support, query_text, query_label, query_set_df, support_set_df = data.generate_test_data(sample_per_class=100)
    query_set_df.to_csv(os.path.join(test_dir, "query_set.csv"), index=False)
    support_set_df.to_csv(os.path.join(test_dir, "support_set.csv"), index=False)
    save_list(support, os.path.join(test_dir, "support_text"))
    save_list(query_text, os.path.join(test_dir, "query_text"))
    save_list(query_label, os.path.join(test_dir, "query_label"))


if __name__ == '__main__':
    print("数据处理")
    exit(0)
    generate_data(input_csv=r"/home/bert_few_shot/data/online_shopping_10_cats.csv",
                  output_dir=r"/home/bert_few_shot/data/output")

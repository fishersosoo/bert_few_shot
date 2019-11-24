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


class OnlineShoppingData():

    def params(self):
        return {"k": self.k, }

    @classmethod
    def _process_dataset(cls, path):
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

    def read_info(self, info_fp):
        with open(info_fp, "r", encoding="utf-8") as fd:
            saved_dict = json.load(fd)
            self.training_set_class = pd.DataFrame(saved_dict["training_set_class"])
            self.k = saved_dict["k"]

    def save_info(self, fp):
        save_dict = dict()
        save_dict["k"] = self.k
        save_dict["training_set_class"] = self.training_set_class.to_dict()
        with open(fp, "w", encoding="utf-8") as fd:
            json.dump(save_dict, fd)

    def __init__(self, input_csv, k=5, info_fp=None):
        """

        Args:
            input_csv:
            k:
            query_size_per_class:
            info_fp:
        """
        log.info("{num} examples per training iter".format(num=k + 1))
        self.input_csv = input_csv
        self.dataset, self.class_info = self._process_dataset(input_csv)
        self.k = k
        self.training_set_class = None
        pd.DataFrame().to_dict()
        if info_fp is not None:
            self.read_info(info_fp)

    def choose_train_test_class(self, training_set_class_num):
        if self.training_set_class is None:
            log.info(
                "No saved train test split info. select {training_set_class_num} training classes randomly.".format(
                    training_set_class_num=training_set_class_num))
            self.training_set_class = self.class_info.sample(training_set_class_num)
        else:
            log.info(
                "Last train test split result is used.".format(
                    training_set_class_num=training_set_class_num))
        self.training_set = self.dataset[self.dataset["class"].isin(self.training_set_class["class"])]
        self.test_set_class = self.class_info[
            np.logical_not(self.class_info["class"].isin(self.training_set_class["class"]))]
        self.test_set = self.dataset[np.logical_not(self.dataset["class"].isin(self.training_set_class["class"]))]

    def generate_training_data(self, training_iter_num, pos_rate=0.5):
        """
        从训练集中随机选取一个类，从类中选取k个样本作为support set

        再以50%概率选取同类样本作为query sample

        Args:
            pos_rate: 正样本比例
            training_iter_num: 生成的样本数量

        Returns:
            support_set_text. string array with shape [training_iter_num, k]
            query_set_text. string array with shape [training_iter_num]
            query_set_label. int array with shape [training_iter_num]
        """
        support_set_text = []
        query_set_text = []
        query_set_label = []
        for training_iter in range(training_iter_num):
            selected_class = self.training_set_class.sample(1)["class"].values[0]
            iter_support_set_text = self.training_set[self.training_set["class"] == selected_class].sample(self.k)[
                "review"].to_list()
            support_set_text.append(iter_support_set_text)

            is_positive = np.random.rand() > pos_rate
            if is_positive:
                query_text = self.training_set[self.training_set["class"] == selected_class].sample(1)["review"].values[0]
            else:
                query_text = self.training_set[self.training_set["class"] != selected_class].sample(1)["review"].values[0]
            query_set_text.append(query_text)
            query_set_label.append(int(is_positive))
        return support_set_text, query_set_text, query_set_label

    def generate_test_data(self, sample_per_class):
        """
        生成测试数据
        先从测试集中每类选取sample_per_class个样本，然后再从每个测试类中选取k个样本作为支撑集。
        Args:
            sample_per_class:

        Returns:
            query_input, string array with shape [run_size]
            support_input. string array with shape [run_size, k]
            query_set_df. DataFrame with column ["cat", "label", "review", "class","class_index"]
                class_index, range from 0 to class_num.
            support_set_df. DataFrame with column ["cat", "label", "review", "class"]
        """
        class_num = len(self.test_set_class["class"])
        query_set_df = None
        support_set_df = None
        support_set_array = []  # string arrat with shape [class_num, k]
        # select test instances and support set
        for class_index, one_class in enumerate(self.test_set_class["class"]):
            class_sample = self.test_set[self.test_set["class"] == one_class]
            # get query sample
            if len(class_sample) < sample_per_class:
                selected_sample = class_sample
            else:
                selected_sample = class_sample.sample(sample_per_class)
            selected_sample["class_index"] = class_index
            if query_set_df is None:
                query_set_df = selected_sample
            else:
                query_set_df = query_set_df.append(selected_sample, ignore_index=True)
            # select support set
            class_support_sample = class_sample.sample(self.k)
            if support_set_df is None:
                support_set_df = class_support_sample
            else:
                support_set_df = support_set_df.append(class_support_sample, ignore_index=True)
            support_set_array.append(class_support_sample["review"].to_list())
        # build input for each test instance
        # each run input k support examples and a test instance
        # there is a support_input array with shape [run_size, k]
        # run_size = class_num * len(query_set_df)
        query_input = []  # string array with shape [run_size]
        support_input = []  # string array with shape [run_size, k]
        for _, one_query in query_set_df.iterrows():
            query_text = one_query["review"]
            for class_index in range(class_num):
                query_input.append(query_text)
                support_input.append(support_set_array[class_index])
        return query_input, support_input, query_set_df, support_set_df


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
    if os.path.exists(os.path.join(output_dir, "data_param.json")):
        info_path = os.path.join(output_dir, "data_param.json")
    else:
        info_path = None
    data = OnlineShoppingData(input_csv, k=5, info_fp=info_path)
    data.choose_train_test_class(15)
    data.save_info(os.path.join(output_dir, "data_param.json"))
    support_set_text, query_set_text, query_set_label = data.generate_training_data(training_iter_num=10000)
    check_dir(training_dir)
    save_list(support_set_text, os.path.join(training_dir, "support_text"))
    save_list(query_set_text, os.path.join(training_dir, "query_text"))
    save_list(query_set_label, os.path.join(training_dir, "query_label"))
    test_dir = os.path.join(output_dir, "predict")
    check_dir(test_dir)
    query_input, support_input, query_set_df, support_set_df = data.generate_test_data(sample_per_class=300)
    query_set_df.to_csv(os.path.join(test_dir, "query_set.csv"), index=False)
    support_set_df.to_csv(os.path.join(test_dir, "support_set.csv"), index=False)
    save_list(support_input, os.path.join(test_dir, "support_text"))
    save_list(query_input, os.path.join(test_dir, "query_text"))


if __name__ == '__main__':
    generate_data(input_csv=r"/home/bert_few_shot/data/online_shopping_10_cats.csv",
                  output_dir=r"/home/bert_few_shot/data/output")

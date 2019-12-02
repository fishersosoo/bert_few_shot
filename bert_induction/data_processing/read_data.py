import json
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("data_process")
# sh = logging.StreamHandler()
# sh.setFormatter(format_str)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


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
                query_text = self.training_set[self.training_set["class"] == selected_class].sample(1)["review"].values[
                    0]
            else:
                query_text = self.training_set[self.training_set["class"] != selected_class].sample(1)["review"].values[
                    0]
            query_set_text.append(query_text)
            query_set_label.append(int(is_positive))
        return support_set_text, query_set_text, query_set_label

    def generate_self_support(self, max_sample_per_class):
        """
        生成测试数据。数据中每个query都是从support里面选取的，所以所有的label都是1
        每个类选择k个support，再将每个support作为query
        Args:
            sample_per_class:

        Returns:
            query_input, string array with shape [run_size]
            support_input. string array with shape [run_size, k]
            query_set_df. DataFrame with column ["cat", "label", "review", "class","class_index"]
                class_index, range from 0 to class_num.
            support_set_df.The same as query_set_df
        """
        query_input=[]
        support_input=[]
        query_set_df=None
        class_num = len(self.test_set_class["class"])
        for class_index, one_class in enumerate(self.test_set_class["class"]):
            class_sample = self.test_set[self.test_set["class"] == one_class]
            for i in range(0,max_sample_per_class,self.k):
                # select k samples
                selected_sample=class_sample.sample(self.k)
                selected_sample["class_index"] = class_index
                if query_set_df is None:
                    query_set_df = selected_sample
                else:
                    query_set_df = query_set_df.append(selected_sample, ignore_index=True)
                for _ ,sample in selected_sample.iterrows():
                    # for each sample build support input use k selected samples
                    support_input.append(selected_sample["review"].tolist())
                    query_input.append(sample["review"])
        return query_input, support_input, query_set_df,query_set_df


    def generate_test_data(self, sample_per_class):
        """
        生成测试数据
        先从测试集中每类选取sample_per_class个样本，对于每个query，随机从样本中选取k个作为support set。
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
        query_input = []  # string array with shape [run_size]
        support_input = []  # string array with shape [run_size, k]
        classes = self.test_set_class["class"].tolist()
        # select test instances and support set
        for class_index, one_class in enumerate(classes):
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
            for _, one_query in selected_sample.iterrows():
                for class_id in range(class_num):
                    # build input for each test instance
                    # each run input k support examples and a test instance
                    # there is a support_input array with shape [run_size, k]
                    # run_size = class_num * len(query_set_df)

                    # select support set
                    class_support_sample = self.test_set[self.test_set["class"] == classes[class_id]].sample(self.k)
                    if support_set_df is None:
                        support_set_df = class_support_sample
                    else:
                        support_set_df = support_set_df.append(class_support_sample, ignore_index=True)

                    support_input.append(class_support_sample["review"].to_list())
                    query_input.append(one_query["review"])

        return query_input, support_input, query_set_df, support_set_df


class TNEWSData():
    @classmethod
    def _process_dataset(cls, path):
        with open(path, encoding="UTF-8") as source_file:
            dataset = {"sample_id": [], "class_id": [], "class_name": [], "text": []}
            for line in source_file:
                sample_id, class_id, class_name, text, tag = line.split("_!_")
                dataset["sample_id"].append(int(sample_id))
                dataset["class_id"].append(int(class_id))
                dataset["class_name"].append(class_name)
                dataset["text"].append(text)


class NSData():
    @classmethod
    def _process_dataset(cls, path):

        with open(path, encoding="utf-8") as f:
            guide_df = {"guide_id": [], "label": [], "title": []}
            guides = json.load(f)
            for guide in guides:
                guide_df["guide_id"].append(guide["guide_id"])
                guide_df["title"].append(guide["title"])
                guide_df["label"].append(guide["label"])
            return pd.DataFrame(guide_df)

    def __init__(self, path, k=5):
        self.guides = self._process_dataset(path)
        self.labels = []
        for i, guide in self.guides.iterrows():
            self.labels.extend(guide["label"])
        self.labels = set(self.labels)
        self.k = k

    def generate_test_data(self, output_dir):
        """

        Returns:
            query_input, string array with shape [run_size]
            support_input. string array with shape [run_size, k]
            query_set_df. DataFrame with column ["title", 0, 1, 2, ... , class_num]
            class_index, class_id to class_name
            support_set_df. DataFrame with column ["title", "labels"]
        """
        query_input = []
        support_input = []
        predict_labels = []
        support_set_texts = []  # [class_num, k]
        df = self.guides[["title"]].copy()
        for label in self.labels:
            has_label = self.guides["label"].apply(lambda labels: label in labels)
            count = np.sum(has_label)
            log.info("{label}:{count}".format(label=label, count=count))
            if count > 25:
                predict_labels.append(label)
        log.info("label to predict: {label}".format(label=predict_labels))
        class_num = len(predict_labels)
        for label_id, label in enumerate(predict_labels):
            has_label = self.guides["label"].apply(lambda labels: label in labels)
            df[label_id] = has_label.astype(int)
            support_set_text = self.guides[has_label].sample(self.k)["title"].to_list()
            support_set_texts.append(support_set_text)
        support_set_texts_flattent = np.array(support_set_texts).reshape(-1)
        query_set_df = df[np.logical_not(df["title"].isin(support_set_texts_flattent))]
        for query in query_set_df["title"]:
            for class_id in range(class_num):
                query_input.append(query)
                support_input.append(support_set_texts[class_id])
        info_dict = {"predict_labels": predict_labels, "query_set_df": query_set_df.to_dict()}
        with open(os.path.join(output_dir, "info.json"), 'w', encoding='utf-8') as f:
            json.dump(info_dict, f)
        np.save(os.path.join(output_dir, "support_text"), np.array(support_input))
        np.save(os.path.join(output_dir, "query_text"), np.array(query_input))


if __name__ == '__main__':
    d = NSData("/home/bert_few_shot/data/source/NS/source.json", 5)
    d.generate_test_data("/home/bert_few_shot/data/output/NS/predict")

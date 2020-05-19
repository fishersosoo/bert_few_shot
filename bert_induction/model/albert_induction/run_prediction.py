# coding=utf-8
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from model.albert_induction.classifier import Classifier

flags = tf.flags

FLAGS = flags.FLAGS


def eval(ret_path, class_info_path):
    class_info = pd.read_csv(class_info_path,
                             index_col=None, encoding="utf-8")
    ret_df = pd.read_csv(ret_path,
                         index_col=None, encoding="utf-8")
    test_classes = class_info[class_info["for_training"] == 0]["class"]
    score_col = ["class_" + str(class_id) for class_id in test_classes]
    ret_df["predict"] = np.argmax(ret_df[score_col].values, axis=1)
    ret_df["predict"] = ret_df["predict"].map(lambda index: test_classes.tolist()[index])
    df=pd.DataFrame()
    df["real"]=ret_df["class"]
    df["predict"] =ret_df["predict"]
    df["true"]=(ret_df["predict"] == ret_df["class"])
    print(df)
    print(sum(ret_df["predict"] == ret_df["class"]))

def test_eval():
    class_info="/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/class_info.csv"
    ret_path="/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/result/large_example.csv"
    eval(ret_path,class_info)

def test_predict():
    test_df = pd.read_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/test.csv",
                          index_col=None, encoding="utf-8")
    class_info = pd.read_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/class_info.csv",
                             index_col=None, encoding="utf-8")
    classifier = Classifier()
    classifier.load("/home/bert_few_shot/models/trained/test_1")

    test_classes = class_info[class_info["for_training"] == 0]["class"]

    query = test_df.sample(5)
    class_support = dict()
    for test_class_id in test_classes:
        class_support[test_class_id] = test_df[test_df["class"] == test_class_id]["review"].sample(5).tolist()
        ret = classifier.predict(class_support[test_class_id], query["review"].tolist())
        query["class_" + str(test_class_id)] = ret["score"]
    query.to_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/result/small_example.csv", index=False,
                 encoding="utf-8")

    query = test_df.sample(30)
    class_support = dict()
    for test_class_id in test_classes:
        class_support[test_class_id] = test_df[test_df["class"] == test_class_id]["review"].sample(5).tolist()
        ret = classifier.predict(class_support[test_class_id], query["review"].tolist())
        query["class_" + str(test_class_id)] = ret["score"]
    query.to_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/result/large_example.csv", index=False,
                 encoding="utf-8")

    # query = test_df
    # class_support = dict()
    # for test_class_id in test_classes:
    #     class_support[test_class_id] = test_df[test_df["class"] == test_class_id]["review"].sample(5).tolist()
    #     ret = classifier.predict(class_support[test_class_id], query["review"].tolist())
    #     query["class_" + str(test_class_id)] = ret["score"]
    # query.to_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/result/large_example.csv", index=False,
    #              encoding="utf-8")


# def main(_):
#     # 训练参数设置
#     test_df = pd.read_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/test.csv",
#                           index_col=None, encoding="utf-8")
#     class_info = pd.read_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/class_info.csv",
#                              index_col=None, encoding="utf-8")
#     classifier = Classifier()
#     classifier.load("/home/bert_few_shot/models/trained/test_1")
#     classifier.predict()


if __name__ == '__main__':
    # test_predict()

    test_eval()
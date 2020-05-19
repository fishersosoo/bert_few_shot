# coding=utf-8
import os
import sklearn.metrics
import tensorflow as tf
import pandas as pd
import numpy as np
from model.albert_induction.classifier import Classifier
import time

flags = tf.flags
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(60)


def eval(ret_df, class_info):
    test_classes = class_info[class_info["for_training"] == 0]["class"]
    score_col = ["class_" + str(class_id) for class_id in test_classes]
    ret_df["predict"] = np.argmax(ret_df[score_col].values, axis=1)
    ret_df["predict"] = ret_df["predict"].map(lambda index: test_classes.tolist()[index])
    return ret_df


def predict(model_dir, k, batch_size=50):
    test_df = pd.read_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/test.csv",
                          index_col=None, encoding="utf-8")
    class_info = pd.read_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/class_info.csv",
                             index_col=None, encoding="utf-8")

    classifier = Classifier()
    classifier.load(model_dir)

    test_classes = class_info[class_info["for_training"] == 0]["class"]

    cur = 0
    result_df = pd.DataFrame()
    batch_num = int(np.ceil(len(test_df) / batch_size))
    print("*************************")
    print("batch num = " + str(batch_num))
    print("batch size = " + str(batch_size))
    print("k = " + str(k))
    print("***********RUNNING*******")

    start = time.time()
    while cur < len(test_df):
        if int(cur / batch_size) % int(batch_num / 10) == 0 and cur != 0:
            print("{percent:.2%} {QPS} QPS".format(percent=cur / batch_size / batch_num,
                                                   QPS=int(
                                                       cur * len(test_classes) * batch_size / (time.time() - start))))
        query = test_df[cur:cur + batch_size]
        cur += batch_size
        class_support = dict()
        for test_class_id in test_classes:
            class_support[test_class_id] = test_df[test_df["class"] == test_class_id]["review"].sample(k).tolist()
            ret = classifier.predict(class_support[test_class_id], query["review"].tolist())
            query["class_" + str(test_class_id)] = ret["score"]
        result_df = result_df.append(query, ignore_index=True)
    print("***********DONE**********")
    return result_df, class_info


def predict_and_eval(k=10, batch_size=20):
    model_dir = "/home/bert_few_shot/models/trained/test_1"
    output_dir = "/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/result"
    result, class_info = predict(model_dir, k, batch_size)
    result = eval(result, class_info)
    result.to_csv((os.path.join(output_dir, 'result.csv')), encoding='utf-8', index=False)
    y_true = result["class"]
    y_pred = result["predict"]
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
    precision = sklearn.metrics.precision_score(y_true, y_pred, average='micro')
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='micro')
    print("*************************")
    summary = "k:\t{k}\nf1:\t{f1}\nprecision:\t{precision}\nrecall:\t{recall}".format(k=k, f1=f1, precision=precision,
                                                                                      recall=recall)
    print("*************************")
    print(summary)
    with open(os.path.join(output_dir, 'result.txt'), encoding='utf-8', mode='w') as summary_fd:
        summary_fd.write(summary)
    return precision, recall, f1


if __name__ == '__main__':
    tf.logging.set_verbosity(60)
    data = {"k": [], "precision": [], "recall": [], "f1": []}
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    for k in range(5, 55, 5):
        for i in range(10):
            precision, recall, f1 = predict_and_eval(k, batch_size=50)
            data["k"].append(k)
            data["precision"].append(precision)
            data["recall"].append(recall)
            data["f1"].append(f1)
    pd.DataFrame(data).to_csv("/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot/k_precision_cur.csv",
                              encoding='utf-8', index=False)

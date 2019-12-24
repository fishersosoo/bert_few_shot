# coding=utf-8
import os
import logging
import numpy as np

from data_processing.read_data import log
from data_processing import OnlineShoppingData


def conver_to_onehot(y, class_num):
    return np.eye(class_num)[y.reshape(-1)].astype(int)


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
    query_input, support_input, query_set_df, support_set_df = data.generate_self_support(300)
    query_set_df.to_csv(os.path.join(test_dir, "query_set.csv"), index=False)
    support_set_df.to_csv(os.path.join(test_dir, "support_set.csv"), index=False)
    save_list(support_input, os.path.join(test_dir, "support_text"))
    save_list(query_input, os.path.join(test_dir, "query_text"))



if __name__ == '__main__':
    generate_data(input_csv=r"/home/bert_few_shot/data/online_shopping_10_cats.csv",
                  output_dir=r"/home/bert_few_shot/data/output/self_support")

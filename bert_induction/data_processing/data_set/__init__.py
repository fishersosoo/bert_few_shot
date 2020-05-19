from abc import abstractmethod
import logging
import requests
import json
import re
import time

log = logging.getLogger("data_process")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


class Dataset(object):
    """数据基类。对象不保存任何信息，是无状态的。"""
    @abstractmethod
    def build_train_test(self, source, output_dir,**kwargs):
        """        读取原始文件        """
        raise NotImplementedError()

    # @abstractmethod
    # def read_raw_data_file(self, raw_fp, *args, **kwargs):
    #     """        读取原始文件        """
    #     raise NotImplementedError()

    # @abstractmethod
    # def train_test_split(self, *args, **kwargs):
    #     """划分训练集和测试集"""
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def get_training_examples(self, *args, **kwargs):
    #     """获取训练样本"""
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def get_test_examples(self, *args, **kwargs):
    #     """获取测试样本"""
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def convert_examples_to_features(self, examples, label_list, params,
    #                                  tokenizer):
    #     """
    #     构建成features，配合build_input_fn使用
    #     """
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def build_input_fn(self, *args, **kwargs):
    #     raise NotImplementedError()


split_symbol = re.compile("。！？!?")


def get_entities(sentence):
    max_qph = 5000  # 每小时访问次数限制
    min_delay = 3600 / max_qph
    start = time.time()
    api_key = "bbfb6cb5a5446b0ec3647d442e7a14a1"
    api = "http://shuyantech.com/api/entitylinking/cutsegment"
    payload = {'q': sentence, "apikey": api_key}
    r = requests.get(api, params=payload)
    run_time = time.time() - start
    # if run_time < min_delay:
    #     time.sleep(min_delay - run_time)
    return r.text


if __name__ == '__main__':
    text = "测试"
    start=time.time()
    for i in  range(10):
        get_entities(text)
    end=time.time()
    print(end-start)


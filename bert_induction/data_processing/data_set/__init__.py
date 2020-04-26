from abc import abstractmethod
import logging

log = logging.getLogger("data_process")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


class Dataset(object):
    """数据基类。对象不保存任何信息，是无状态的。"""

    @abstractmethod
    def read_raw_data_file(self, raw_fp, *args, **kwargs):
        """        读取原始文件        """
        raise NotImplementedError()

    @abstractmethod
    def train_test_split(self, *args, **kwargs):
        """划分训练集和测试集"""
        raise NotImplementedError()

    @abstractmethod
    def get_training_examples(self, *args, **kwargs):
        """获取训练样本"""
        raise NotImplementedError()

    @abstractmethod
    def get_test_examples(self, *args, **kwargs):
        """获取测试样本"""
        raise NotImplementedError()

    @abstractmethod
    def convert_examples_to_features(self, examples, label_list, params,
                                     tokenizer):
        """
        构建成features，配合build_input_fn使用
        """
        raise NotImplementedError()

    @abstractmethod
    def build_input_fn(self, *args, **kwargs):
        raise NotImplementedError()

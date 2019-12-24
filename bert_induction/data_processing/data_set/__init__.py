from abc import abstractmethod
import logging

log = logging.getLogger("data_process")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


class Dataset(object):
    @abstractmethod
    def read_raw_data_file(self, raw_fp, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def train_test_split(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_training_examples(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_test_examples(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def write_example(self, *args, **kwargs):
        """
        将输入数据转化成TFRecord文件
        每一个 training 迭代写成一个example
        Args:
            fp_in:
            fp_out:
            max_seq_length:
            tokenizer:
            do_predict:

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def build_file_base_input_fn(self, *args, **kwargs):
        """
        读取TF_Record构建输入pipeline
        Args:
            input_file:
            params:
            is_training:
            drop_remainder:

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def convert_examples_to_features(self, examples, label_list, params,
                                     tokenizer):
        """
        构建成features，配合build_input_fn使用
        Args:
            examples:
            label_list:
            params:
            tokenizer:

        Returns:

        """
        raise NotImplementedError()

    @abstractmethod
    def build_input_fn(self, *args, **kwargs):
        raise NotImplementedError()

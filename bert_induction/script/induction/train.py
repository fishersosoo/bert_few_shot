# coding=utf-8
import tensorflow as tf

from data_processing import SegmentFixedTokenizer, get_dataset_cls

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", None, "配置文件路径")
flags.DEFINE_bool("generate_data", True, "生成测试数据和训练数据")
flags.DEFINE_string("raw_data_path", None, "原始数据文件路径")
flags.DEFINE_string("data_dir", None, "数据文件所在目录")
flags.DEFINE_string("data_set",None,"使用的数据集解析类")
flags.DEFINE_string("tokenizer_path",None,"")
flags.DEFINE_string("tokenizer_dict",None,"")


def main():
    data_set_cls = get_dataset_cls(FLAGS.data_set)
    tokenizer = SegmentFixedTokenizer(path=FLAGS.tokenizer_path, dict_path=FLAGS.tokenizer_dict)

    pass


if __name__ == '__main__':
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("data_dir")
    tf.app.run()

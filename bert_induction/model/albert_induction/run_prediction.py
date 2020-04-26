# coding=utf-8
import os
import tensorflow as tf
from model.induction.classifier import Classifier

flags = tf.flags

FLAGS = flags.FLAGS
# classifier params
flags.DEFINE_string("data_dir", None, "输入数据目录")
flags.DEFINE_string("model_config", None, "模型配置文件目录")
flags.DEFINE_string("tokenizer", "segment_fixed_tokenizer", "tokenizer name")
flags.DEFINE_string("tokenizer_dir", None, "")
flags.DEFINE_string("output_dir", None, "结果输出目录")

# prediction params
flags.DEFINE_string("data_set", None, "data set name")
flags.DEFINE_string("model_dir", None, "model saved dir")
flags.DEFINE_integer("max_seq_length", 64, "max seq length")
flags.DEFINE_string("init_checkpoint", None, "checkpoint path")
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("predict_class_num", 5, "predict_class_num")
flags.DEFINE_bool("use_exist_examples", False, "use_exist_examples")


def main(_):
    # 训练参数设置
    data_dir = FLAGS.data_dir
    classifier = Classifier(model_config_path=FLAGS.model_config,
                            tokenizer=FLAGS.tokenizer,
                            output_dir=FLAGS.output_dir,
                            tokenizer_dir=FLAGS.tokenizer_dir)
    classifier.predict(example_fp=os.path.join(data_dir, "test_examples.npy"),
                       data_set=FLAGS.data_set,
                       max_seq_length=FLAGS.max_seq_length,
                       init_checkpoint=FLAGS.init_checkpoint,
                       predict_class_num=FLAGS.predict_class_num,
                       output_dir=FLAGS.output_dir,
                       batch_size=FLAGS.batch_size
                       )



if __name__ == '__main__':
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("model_config")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("data_set")
    flags.mark_flag_as_required("predict_class_num")
    tf.app.run()

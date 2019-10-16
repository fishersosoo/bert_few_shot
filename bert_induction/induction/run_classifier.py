# encoding=utf-8
import tensorflow as tf

from induction.modeling import classifier

flags = tf.flags

FLAGS = flags.FLAGS
# 脚本参数定义
flags.DEFINE_string("vocab_file", None, "字典路径")
flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_string("data_dir", None, "输入数据所在目录")
flags.DEFINE_string("config_file", None, "配置文件路径")
flags.DEFINE_string("bert_config_file", None, "配置文件路径")
flags.DEFINE_string("output_dir", None, "输出目录（模型checkpoints、训练数据会被写到该目录下）")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for. "
                                             "E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

# tpu相关参数
tf.flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
tf.flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name "
                                         "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
                                         "url.")
tf.flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not "
                                         "specified, we will attempt to automatically detect the GCE project from "
                                         "metadata.")
tf.flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not "
                                            "specified, we will attempt to automatically detect the GCE project from "
                                            "metadata.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
tf.flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def main():
    classifier(FLAGS.vocab_file,
               FLAGS.data_dir,
               FLAGS.config_file,
               FLAGS.bert_config_file,
               FLAGS.output_dir,
               FLAGS.max_seq_length,
               FLAGS.init_checkpoint,
               FLAGS.do_train,
               FLAGS.do_eval,
               FLAGS.do_predict,
               FLAGS.train_batch_size,
               FLAGS.eval_batch_size,
               FLAGS.predict_batch_size,
               FLAGS.num_train_epochs,
               FLAGS.warmup_proportion,
               FLAGS.save_checkpoints_steps,
               FLAGS.iterations_per_loop,
               FLAGS.learning_rate,
               FLAGS.use_tpu, FLAGS.tpu_name,
               FLAGS.tpu_zone, FLAGS.gcp_project,
               FLAGS.master, FLAGS.num_tpu_cores
               )


if __name__ == '__main__':
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("config_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

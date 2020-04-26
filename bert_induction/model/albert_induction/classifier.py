# encoding=utf-8
"""运行配置，管理输入输出"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from data_processing import get_tokenizer_cls, get_dataset_cls
from model.induction.config import ModelConfig
from model.induction.modeling import model_fn_builder


class Classifier(object):
    def __init__(self,
                 model_config_path,
                 tokenizer,
                 output_dir,
                 tokenizer_dir,
                 save_checkpoints_steps=5000):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.tokenizer = get_tokenizer_cls(tokenizer)(
            os.path.join(tokenizer_dir, "merge_sgns_bigram_char300.txt"),
            os.path.join(tokenizer_dir, "user_dict.txt")
        )

        tf.gfile.MakeDirs(output_dir)
        self.output_dir = output_dir
        self.model_config, self.run_config = config_setup(model_config_path,
                                                          output_dir,
                                                          save_checkpoints_steps)

    def train(self,
              example_fp,
              data_set,
              model_dir=None,
              max_seq_length=32,
              init_checkpoint=None,
              batch_size=32,
              num_train_epochs=3.0,
              learning_rate=5e-5,
              warmup_proportion=0.1,
              use_exist_examples=False):

        train_examples = example_fp
        train_record = os.path.join(self.output_dir, "train.tf_record")
        data_set = get_dataset_cls(data_set)()
        example_nums = data_set.write_example(
            fp_in=train_examples,
            fp_out=train_record,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,use_exist=use_exist_examples)
        num_train_steps = int(
            example_nums / batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        train_input_fn = data_set.build_file_base_input_fn(input_file=train_record,
                                                           model_config=self.model_config,
                                                           batch_size=batch_size,
                                                           max_seq_length=max_seq_length,
                                                           is_training=True)
        if model_dir is None:
            model_dir = self.output_dir
            run_config=self.run_config
        else:
            run_config=self.run_config.replace(model_dir=model_dir)
        model_fn = model_fn_builder(config=self.model_config,
                                    init_checkpoint=init_checkpoint,
                                    max_seq_length=max_seq_length,
                                    learning_rate=learning_rate,
                                    num_train_steps=num_train_steps,
                                    num_warmup_steps=num_warmup_steps)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            config=run_config)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", example_nums)
        tf.logging.info("  Batch size = %d", batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        estimator.train(input_fn=train_input_fn, steps=num_train_steps)

    def eval(self, example_fp, max_seq_length, init_checkpoint, data_set, batch_size):
        eval_examples = example_fp
        eval_file = os.path.join(self.output_dir, "eval.tf_record")
        data_set = get_dataset_cls(data_set)()
        eval_examples_num = data_set.write_example(eval_examples,
                                                   eval_file,
                                                   max_seq_length=max_seq_length,
                                                   do_predict=False,
                                                   tokenizer=self.tokenizer)
        model_fn = model_fn_builder(config=self.model_config,
                                    init_checkpoint=init_checkpoint,
                                    max_seq_length=max_seq_length)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=self.output_dir,
            config=self.run_config)
        data_set = get_dataset_cls(data_set)()

        eval_input_fn = data_set.build_file_base_input_fn(input_file=os.path.join(example_fp, "eval.tf_record"),
                                                          model_config=self.model_config,
                                                          batch_size=batch_size,
                                                          max_seq_length=max_seq_length,
                                                          is_training=False)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)
        # 输出结果
        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def predict(self, example_fp, data_set,
                max_seq_length, predict_class_num, init_checkpoint,
                output_dir, batch_size):
        predict_examples = example_fp
        predict_file = os.path.join(example_fp, "predict.tf_record")
        data_set = get_dataset_cls(data_set)()

        predict_iter_num = data_set.write_example(
            predict_examples,
            predict_file,
            do_predict=True,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer
        )
        model_fn = model_fn_builder(
            config=self.model_config,
            init_checkpoint=init_checkpoint,
            max_seq_length=max_seq_length)
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=self.run_config
        )
        predict_input_fn = data_set.build_file_base_input_fn(
            input_file=predict_file,
            model_config=self.model_config,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            is_training=False,
            drop_remainder=False
        )
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num iter = %d ", predict_iter_num)
        tf.logging.info("  Batch size = %d", batch_size)
        prediction = estimator.predict(input_fn=predict_input_fn)
        results = [one for one in prediction]
        result_df = parse_result(results, self.model_config, predict_class_num)
        output_predict_file = os.path.join(output_dir, "predict_results.csv")
        result_df.to_csv(output_predict_file, encoding="utf-8", index=False)


def parse_result(results, model_config, predict_class_num):
    iter_per_num = int(np.ceil(predict_class_num / model_config.c))
    iter_num = len(results)
    sample_num = iter_num * model_config.query_size / iter_per_num
    result_data = {"sample_id": [], "predict": []}
    for class_id in range(predict_class_num):
        result_data[str(class_id)] = []
    for sample_id in range(sample_num):
        run_id = int(sample_id / model_config.query_size)
        sample_index = sample_id % model_config.query_size
        result_data["sample_id"].append(sample_id)
        scores = []
        for class_id in range(predict_class_num):
            iter_id = run_id * iter_per_num + int(predict_class_num / model_config.c)
            class_index = class_id % model_config.c
            scores.append(results[iter_id]["relation_score"][sample_index][class_index])
        for i, score in enumerate(scores):
            result_data[str(i)].append(score)
        result_data["predict"].append(np.argmax(scores))
    return pd.DataFrame(result_data)


# def save_result(predict_output_dir, predict_class_num, predict_run_num, results):
#     tf.logging.info("result size = %d", len(results))
#     # checker
#     check_query_embedding_equal = CheckEmbedding("query embeddings")
#     check_support_embedding_equal = CheckEmbedding("support embeddings")
#     check_class_vector_equal = CheckEmbedding("class vector")
#     # file path
#     output_predict_file = os.path.join(predict_output_dir, "test_results.csv")
#     output_embeddings_file = os.path.join(predict_output_dir, "test_embeddings.csv")
#     output_class_vector_fp = os.path.join(predict_output_dir, "class_vector.csv")
#     output_support_embeddings_fp = os.path.join(predict_output_dir, "support_embeddings.csv")
#     # data
#     support_embeddings = {"class_id": [], "sample_id": [], "embeddings": []}
#     class_vectors = {"class_id": [], "embeddings": []}
#     # get class vector for debug
#     for class_id in range(predict_class_num):
#         support_embedding = None
#         class_vector = None
#         for run_id in range(class_id, predict_run_num, predict_class_num):
#             if support_embedding is None:
#                 support_embedding = results[run_id]["support_embedding"]
#             else:
#                 check_support_embedding_equal(support_embedding, results[run_id]["support_embedding"])
#             if class_vector is None:
#                 class_vector = results[run_id]["class_vector"]
#             else:
#                 check_class_vector_equal(class_vector, results[run_id]["class_vector"])
#
#         for sample_id, embedding in enumerate(support_embedding):
#             support_embeddings["class_id"].append(class_id)
#             support_embeddings["sample_id"].append(sample_id)
#             support_embeddings["embeddings"].append(embedding.tolist())
#         class_vectors["embeddings"].append(class_vector)
#         class_vectors["class_id"].append(class_id)
#     # write file
#     pd.DataFrame(class_vectors).to_csv(output_class_vector_fp, index=False)
#     pd.DataFrame(support_embeddings).to_csv(output_support_embeddings_fp, index=False)
#     # get query sample result
#     result_data = {"sample_id": [], "prediction": [], "embeddings": []}
#     for class_index in range(predict_class_num):
#         result_data[str(class_index)] = []
#     for sample_id in range(predict_examples_num):
#         result_data["sample_id"].append(sample_id)
#         probabilities = []
#         query_embedding = None
#         for class_id in range(predict_class_num):
#             result_id = sample_id * predict_class_num + class_id
#             class_probability = results[result_id]["relation_score"][0]
#             probabilities.append(class_probability)
#             result_data[str(class_id)].append(class_probability)
#             if query_embedding is None:
#                 query_embedding = results[result_id]["query_embedding"]
#             else:
#                 check_query_embedding_equal(results[result_id]["query_embedding"], query_embedding)
#         result_data["embeddings"].append(query_embedding.tolist())
#         result_data["prediction"].append(np.argmax(probabilities))
#     result_df = pd.DataFrame(result_data)
#     result_df[["sample_id", "embeddings"]].to_csv(output_embeddings_file, index=False)
#     result_df.drop(columns=["embeddings"]).to_csv(output_predict_file, index=False)


def config_setup(config_path, output_dir, save_checkpoints_steps):
    model_config = ModelConfig.from_json_file(config_path)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    return model_config, run_config

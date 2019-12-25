# encoding=utf-8
"""运行配置，管理输入输出"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from data_processing import SegmentFixedTokenizer, get_tokenizer_cls
from model.induction.config import ModelConfig
from model.induction.modeling import write_example, model_fn_builder, file_based_input_fn_builder
from model.common.base_func import CheckEmbedding


def classifier(
        data_dir,
        config_path,
        output_dir,
        tokenizer,
        tokenizer_dir,
        data_set,
        max_seq_length=32,
        init_checkpoint=None,
        do_train=False,
        do_eval=False,
        do_predict=False,
        train_batch_size=32,
        eval_batch_size=8,
        predict_batch_size=8,
        num_train_epochs=3.0,
        warmup_proportion=0.1,
        save_checkpoints_steps=1000,
        learning_rate=5e-5,
        use_tpu=False,
        predict_class_num=5,
):
    """

    Args:
        data_dir:
        config_path:
        output_dir:
        tokenizer:
        tokenizer_dir:
        data_set:
        max_seq_length:
        init_checkpoint:
        do_train:
        do_eval:
        do_predict:
        train_batch_size:
        eval_batch_size:
        predict_batch_size:
        num_train_epochs:
        warmup_proportion:
        save_checkpoints_steps:
        learning_rate:
        use_tpu:
        predict_class_num:

    Returns:

    """
    tf.logging.set_verbosity(tf.logging.INFO)

    if not do_train and not do_eval and not do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    tf.gfile.MakeDirs(output_dir)
    tokenizer = get_tokenizer_cls(tokenizer)(
        os.path.join(tokenizer_dir, "merge_sgns_bigram_char300.txt"),
        os.path.join(tokenizer_dir, "user_dict.txt")
    )

    model_config, run_config = config_setup(config_path, output_dir, save_checkpoints_steps)

    example_nums = None
    num_train_steps = None
    num_warmup_steps = None
    batch_size = 1
    if do_train:
        batch_size = train_batch_size
    if do_predict:
        batch_size = predict_batch_size
    if do_eval:
        batch_size = eval_batch_size

    if do_train:
        example_nums = write_example(os.path.join(data_dir, "train"),
                                     os.path.join(data_dir, "train.tf_record"),
                                     max_seq_length=max_seq_length,
                                     tokenizer=tokenizer)
        num_train_steps = int(
            example_nums / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        train_input_fn = data_set.build_file_base_input_fn(input_file=os.path.join(data_dir, "train.tf_record"),
                                                           config=model_config,
                                                           batch_size=train_batch_size,
                                                           max_seq_length=max_seq_length,
                                                           is_training=True)

    model_fn = model_fn_builder(config=model_config,
                                init_checkpoint=init_checkpoint,
                                max_seq_length=max_seq_length,
                                learning_rate=learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=output_dir,
        config=run_config)

    if do_train:
        train_file = os.path.join(data_dir, "train.tf_record")
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", example_nums)
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(train_file, model_config, max_seq_length, True, True)
        estimator.train(input_fn=train_input_fn, steps=num_train_steps)

    if do_eval:
        eval_steps = None
        eval_file = os.path.join(data_dir, "eval.tf_record")
        eval_examples_num = write_example(os.path.join(data_dir, "eval"),
                                          eval_file,
                                          max_seq_length=max_seq_length,
                                          tokenizer=tokenizer)
        eval_drop_remainder = True if use_tpu else False
        eval_input_fn = file_based_input_fn_builder(eval_file, model_config, max_seq_length, False, eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        # 输出结果
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if do_predict:
        predict_file = os.path.join(data_dir, "predict.tf_record")
        predict_run_num = write_example(os.path.join(data_dir, "predict"),
                                        predict_file, do_predict=True,
                                        max_seq_length=max_seq_length,
                                        tokenizer=tokenizer)
        predict_examples_num = int(predict_run_num / predict_class_num)
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d ", predict_examples_num)
        tf.logging.info("  Num runs = %d ", predict_run_num)
        tf.logging.info("  Batch size = %d", predict_batch_size)
        predict_drop_remainder = True if use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            config=model_config,
            max_seq_length=max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder
        )
        result = [one for one in estimator.predict(input_fn=predict_input_fn)]
        tf.logging.info("result size = %d", len(result))
        # checker
        check_query_embedding_equal = CheckEmbedding("query embeddings")
        check_support_embedding_equal = CheckEmbedding("support embeddings")
        check_class_vector_equal = CheckEmbedding("class vector")
        # file path
        output_predict_file = os.path.join(output_dir, "test_results.csv")
        output_embeddings_file = os.path.join(output_dir, "test_embeddings.csv")
        output_class_vector_fp = os.path.join(output_dir, "class_vector.csv")
        output_support_embeddings_fp = os.path.join(output_dir, "support_embeddings.csv")
        # data
        support_embeddings = {"class_id": [], "sample_id": [], "embeddings": []}
        class_vectors = {"class_id": [], "embeddings": []}
        # get class vector for debug
        for class_id in range(predict_class_num):
            support_embedding = None
            class_vector = None
            for run_id in range(class_id, predict_run_num, predict_class_num):
                if support_embedding is None:
                    support_embedding = result[run_id]["support_embedding"]
                else:
                    check_support_embedding_equal(support_embedding, result[run_id]["support_embedding"])
                if class_vector is None:
                    class_vector = result[run_id]["class_vector"]
                else:
                    check_class_vector_equal(class_vector, result[run_id]["class_vector"])

            for sample_id, embedding in enumerate(support_embedding):
                support_embeddings["class_id"].append(class_id)
                support_embeddings["sample_id"].append(sample_id)
                support_embeddings["embeddings"].append(embedding.tolist())
            class_vectors["embeddings"].append(class_vector)
            class_vectors["class_id"].append(class_id)

        # write file
        pd.DataFrame(class_vectors).to_csv(output_class_vector_fp, index=False)
        pd.DataFrame(support_embeddings).to_csv(output_support_embeddings_fp, index=False)

        # get query sample result
        result_data = {"sample_id": [], "prediction": [], "embeddings": []}
        for class_index in range(predict_class_num):
            result_data[str(class_index)] = []
        for sample_id in range(predict_examples_num):
            result_data["sample_id"].append(sample_id)
            probabilities = []
            query_embedding = None
            for class_id in range(predict_class_num):
                result_id = sample_id * predict_class_num + class_id
                class_probability = result[result_id]["relation_score"][0]
                probabilities.append(class_probability)
                result_data[str(class_id)].append(class_probability)
                if query_embedding is None:
                    query_embedding = result[result_id]["query_embedding"]
                else:
                    check_query_embedding_equal(result[result_id]["query_embedding"], query_embedding)
            result_data["embeddings"].append(query_embedding.tolist())
            result_data["prediction"].append(np.argmax(probabilities))
        result_df = pd.DataFrame(result_data)
        result_df[["sample_id", "embeddings"]].to_csv(output_embeddings_file, index=False)
        result_df.drop(columns=["embeddings"]).to_csv(output_predict_file, index=False)


def config_setup(config_path, output_dir, save_checkpoints_steps):
    model_config = ModelConfig.from_json_file(config_path)
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    return model_config, run_config

# coding=utf-8
import os

from data_processing import get_tokenizer_cls, get_dataset_cls
from model.induction.config import ModelConfig


def main():
    # 训练参数设置
    data_dir = "/home/bert_few_shot/data/online_shopping_10_cats/2-way 5-shot"
    example_path = os.path.join(data_dir, "training_examples.npy")
    config_path = "/home/bert_few_shot/models/trained/induction/config.json"
    tokenizer = "segment_fixed_tokenizer"
    tokenizer_dir = "/home/bert_few_shot/models/vector"
    data_set = "online_shopping"
    max_seq_length = 64
    batch_size=

    #
    model_config = ModelConfig.from_json_file(config_path)
    tokenizer = get_tokenizer_cls(tokenizer)(
        os.path.join(tokenizer_dir, "merge_sgns_bigram_char300.txt"),
        os.path.join(tokenizer_dir, "user_dict.txt")
    )
    data_set = get_dataset_cls(data_set)()
    example_nums=data_set.write_example(fp_in=example_path,
                           fp_out=os.path.join(data_dir, "train.tf_record"),
                           max_seq_length=max_seq_length,
                           tokenizer=tokenizer,
                           do_predict=False)
    num_train_steps = int(
        example_nums / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    input_fn = data_set.build_file_base_input_fn(input_file=os.path.join(data_dir, "train.tf_record"),
                                                 config=model_config,
                                                 max_seq_length=max_seq_length,
                                                 is_training=True)


if __name__ == '__main__':
    main()

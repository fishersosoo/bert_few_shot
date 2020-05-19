# encoding=utf-8
from data_processing import OnlineShoppingData
from data_processing.data_set.online_shopping_data import file_based_convert_examples_to_features_fn
from model.albert_induction.classifier import Classifier
from model.albert_induction.config import ModelConfig as InductionConfig
from model.albert_zh.modeling import BertConfig

if __name__ == '__main__':
    source_path = "/home/bert_few_shot/data/source/online_shopping_10_cats"
    output_data_dir = "/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot"
    bert_config_path = "/home/bert_few_shot/models/albert_large_zh/albert_config_large.json"
    bert_ckpt_path = "/home/bert_few_shot/models/albert_large_zh/albert_model.ckpt"
    save_dir = "/home/bert_few_shot/models/trained/test_2"
    tokenizer_name = "full_tokenizer"
    vocab_path = "/home/bert_few_shot/models/albert_large_zh/vocab.txt"
    max_len = 64
    batch_size = 1
    num_train_epochs = 10.0
    learning_rate = 1e-5
    use_existed=True
    induction_config = InductionConfig(c=2, k=5, query_size=20, h=50)

    data_set = OnlineShoppingData()

    training_fp, test_fp = data_set.build_train_test(source_path, output_data_dir,use_existed=use_existed)

    bert_config = BertConfig.from_json_file(bert_config_path)
    classifier = Classifier()
    classifier.train(training_data_path=training_fp,
                     save_dir=save_dir,
                     tokenizer_name=tokenizer_name,
                     vocab_path=vocab_path,
                     max_len=max_len,
                     file_based_convert_examples_to_features_fn=file_based_convert_examples_to_features_fn,
                     use_existed=use_existed,
                     induction_config=induction_config,
                     bert_config=bert_config,
                     init_checkpoint=bert_ckpt_path,
                     batch_size=batch_size,
                     num_train_epochs=num_train_epochs,
                     learning_rate=learning_rate)
    # classifier.save(save_dir)

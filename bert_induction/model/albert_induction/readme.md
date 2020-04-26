# Induction Network

将原始induction network实现中的编码器替换成albert

## 训练

```bash
python bert_induction/model/induction/run_training.py \
--data_dir=/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot \
--model_config=/home/bert_few_shot/models/trained/induction/config.json \
--output_dir=/home/bert_few_shot/results/online_shopping_10_cats \
--data_set=online_shopping \
--tokenizer=segment_fixed_tokenizer \
--tokenizer_dir=/home/bert_few_shot/models/vector \
--model_dir=/home/bert_few_shot/models/trained/induction \
--max_seq_length=64 \
--batch_size=32 \
--num_train_epochs=5000.0 \
--use_exist_examples=false \
--learning_rate=0.0005 >/home/bert_few_shot/logs/induction_train.log 2>&1 &
```

## 预测

```bash
python bert_induction/model/induction/run_prediction.py \
--data_dir=/home/bert_few_shot/data/online_shopping_10_cats/2-way_5-shot \
--model_config=/home/bert_few_shot/models/trained/induction/config.json \
--output_dir=/home/bert_few_shot/results/online_shopping_10_cats \
--data_set=online_shopping \
--tokenizer=segment_fixed_tokenizer \
--tokenizer_dir=/home/bert_few_shot/models/vector \
--max_seq_length=64 \
--batch_size=1 \
--init_checkpoint=/home/bert_few_shot/models/trained/induction/model.ckpt-187500.data-00000-of-00001 \
--use_exist_examples=false \
--predict_class_num=6 >/home/bert_few_shot/logs/induction_predict.log 2>&1 &
```


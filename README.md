# few shot

## Changelists

### 2019-11-25

- 复现原文实现，使用单层的双向LSTM使用attention机制进行编码
- k=5, c=1, hidden_size=128, attention_size=64, dropout_prob=0.1, max_length=32, batch_size=32, epochs=32

测试效果：

('书籍', 0):0.79
('水果', 1):0.73
('洗发水', 0):0.27
('热水器', 0):0.61
('热水器', 1):0.63

random acc: 0.2
acc: 0.61
kappa: 0.51

## 数据集处理流程

一般过程，是调用`read_raw_data_file`解析数据文件，然后调用`train_test_split`划分测试集和训练集，调用`get_training_examples`和`get_test_examples`获取样本并使用`write_example`保存成TF文件。在训练或者预测的过程中调用`build_file_base_input_fn`构建模型输入。
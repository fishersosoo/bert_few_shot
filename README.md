# 利用ALbert和知识实现小样本短文本分类

## 功能及实现

1. 使用ALbert作为编码器、induction 结构进行类特征提取、NTN作为评分建模。
2. 增加概念编码模块，使用hanlp分词，将分词结果输入到知识库中查找相关上位词，用ALbert进行编码。

## 已知问题

1. 引入知识的第一步就是要从文本中提取mention，使用预训练的深度学习模型固然有很好的准确率，但是只能提取出机构名NT、人名NR、地名NS的有限类型实体。

## 模型

### baseline模型

使用bert的first token编码作为句子编码，使用3循环的induction中提取类编码，使用NTN网络计算相似度

## 性能和准确率

### 性能

| 编码器                                                       | c    | k    | query | batch_size | max_len | h    | 显存    | 模型     |
| ------------------------------------------------------------ | ---- | ---- | ----- | ---------- | ------- | ---- | ------- | -------- |
| [albert_large_zh](https://storage.googleapis.com/albert_zh/albert_large_zh.zip) | 2    | 5    | 20    | 2          | 32      | 50   | 8543MiB | baseline |
|                                                              |      |      |       |            |         |      |         |          |
|                                                              |      |      |       |            |         |      |         |          |



## 数据集处理流程

一般过程，是调用`read_raw_data_file`解析数据文件，然后调用`train_test_split`划分测试集和训练集，调用`get_training_examples`和`get_test_examples`获取样本并使用`write_example`保存成TF文件。在训练或者预测的过程中调用`build_file_base_input_fn`构建模型输入。

https://github.com/AIRobotZhang/STCKA
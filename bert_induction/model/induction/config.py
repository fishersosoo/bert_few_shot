# coding=utf-8

from model.common import BaseModelConfig


class ModelConfig(BaseModelConfig):
    # 模型超参数（需要config文件中配置，和导出的模型一起打包）
    # c: 类别个数
    # k: 每类样本数量
    # seq_len: 每个样本特征数量
    # query_size: query set 样本数量

    def __init__(self, k=5, dropout_prob=.1, embedding_size=300, hidden_size=128, attention_size=64, h=100):
        """
        模型超参数（需要config文件中配置，和导出的模型一起打包）
        Args:

        """
        self.k = k
        self.dropout_prob = dropout_prob
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.h = h

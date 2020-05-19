# coding=utf-8

from model.common import BaseModelConfig


class ModelConfig(BaseModelConfig):
    # 模型超参数（需要config文件中配置，和导出的模型一起打包）
    # c: 类别个数
    # k: 每类样本数量
    # query_size: query set 样本数量
    # embedding_size: 词向量长度
    # hidden_size: LSTM长度
    # attention_size: 注意力机制得到的句向量长度
    # dropout_prob: drop out概率
    # h: NTN维度

    def __init__(self,
                 c=2,
                 k=5,
                 query_size=20,
                 dropout_prob=.1,
                 h=100):
        """
        模型超参数（需要config文件中配置，和导出的模型一起打包）
        """
        self.c = c
        self.k = k
        self.query_size = query_size
        self.dropout_prob = dropout_prob
        self.h = h


def main():
    config = ModelConfig()
    with open("/home/bert_few_shot/models/trained/induction/config.json", "w") as config_fd:
        config_fd.write(config.to_json_string())


if __name__ == '__main__':
    main()
